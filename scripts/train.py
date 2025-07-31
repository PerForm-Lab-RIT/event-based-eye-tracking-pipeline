"""
File: train.py
Author: Viet Nguyen
Date: 2025-05-28

Description: This script is used to train the event-based eye tracking models.
"""


import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import Dict, Callable, Any, List
import jax
import jax.numpy as jnp
import numpy as np
import collections
import matplotlib.pyplot as plt
from io import BytesIO
import imageio
import re
import functools
from functools import partial as bind
import time
import concurrent.futures

from lib.common.logger import Logger, build_logger
from lib.utils import check_vscode_interactive
from lib import print, Checkpoint, load_config_from_yaml, Config, common, when
from lib.replay.replay import Replay, build_stream

from eet.agents.build import build_agent
from eet.envs.build import build_env
from lib.replay import build_replay
from lib.agent.base import BaseInactiveAgent as Agent
from lib.common.distr import StoppableThread, Context
from lib import nn, tree


extend = lambda x: tree.map(lambda x2: np.asarray(x2)[None], x)
takefirst = lambda x: tree.map(lambda x2: x2[0], x)

def run(config: Config):
  logger: Logger = build_logger(config)
  logdir = pathlib.Path(config.logdir)
  step = logger.step
  usage = common.Usage(**config.run.usage)
  train_agg = common.Agg()
  val_agg = common.Agg()
  train_fps = common.FPS()
  infer_fps = common.FPS()
  episode_stats = common.Agg() # temporary stats per episode
  epstats = common.Agg() # the final episode stats to be added to the logger

  batch_steps = (config.batch_length * config.consec_train + config.replay_context) * config.batch_size
  print(f"[run] batch_steps: {batch_steps}", color='red')
  should_log = when.Clock(config.run.log_every)
  should_report = when.Clock(config.run.report_every)
  should_save = when.Clock(config.run.save_every)

  agent: Agent = build_agent(config)
  carry_infer_train = agent.init_infer(1)
  carry_infer_eval = agent.init_infer(1)
  carry_train = [agent.init_train(config.batch_size)]
  report_carry = [agent.init_report(config.batch_size)]
  replay_train: Replay = build_replay(config, "replay", mode="train")
  replay_eval: Replay = build_replay(config, "replay", mode="train") # we use eval in dataset env, so it can be train here
  dataset_train = iter(agent.stream(build_stream(config, replay_train, "train")))
  dataset_eval = iter(agent.stream(build_stream(config, replay_eval, "report")))
  eval_step_stop = False
  train_step_stop = False

  def logfn(tran: Dict[str, np.ndarray]):
    # tran: (...), no batch, no seq length
    tran['is_dataset_first'] and episode_stats.reset()
    episode_stats.add(f'length', 1, agg='sum')

    if tran['is_dataset_last']:
      # print("[logfn] episode end", color='yellow')
      result = episode_stats.result()
      logger.add({
          f'length': result.pop(f'length'),
      }, prefix='episode')
      epstats.add(result)

  def trainfn(tran, worker):
    needed = config.batch_size * (config.batch_length + config.replay_context) * config.consec_train
    if len(replay_train) < needed:
      # print(f"Replay buffer does not have enough experiences (replay length = {len(replay_train)} < {needed}), skipping training", color='yellow', end='\r')
      return False

    # Perform one training step with normal dataset
    with common.timer.section('stream_next'):
      batch = next(dataset_train) # Half train with normal dataset
      # print(f"[run.trainfn] batch: {batch.keys()}")
    carry_train[0], outs_train, train_mets = agent.train(carry_train[0], batch)
    if 'replay' in outs_train:
      replay_train.update(outs_train['replay'])
    train_agg.add({**train_mets}, prefix='train')
    train_fps.step(batch_steps)
    return True

  def reportfn(tran, worker):
    report_agg = common.Agg()
    for _ in range(config.run.report_batches):
      batch = next(dataset_eval)
      report_carry[0], outs_report, mets = agent.report(report_carry[0], batch)
      if 'replay' in outs_report:
        replay_eval.update(outs_report['replay'])
      report_agg.add(mets)
    logger.add(report_agg.result(), prefix='report')

  def preprocess_observation(obs: Dict[str, np.ndarray]):
    proc = extend(obs)
    # print(jax.tree.map(lambda x: x.shape, proc), color='blue')
    # return nn.cast(proc)
    return proc

  def post_process_outs(outs: Dict[str, np.ndarray]):
    # return tree.map(lambda x: x[0], outs)
    return takefirst(outs)

  # have a separate threads/processes, that do env step here
  # envs = [build_env(config) for _ in range(config.n_envs)]

  # def step_env_with_add(context: Context, replay, env, env_id):
  #   while context.running:
  #     transition = env.step()
  #     replay.add(transition, env_id)

  # Use StoppableThread for each environment
  # env_threads: List[StoppableThread] = []
  # for i, env in enumerate(envs):
  #   t = StoppableThread(step_env_with_add, replay_train, env, i, name=f"env{i}-step", start=True)
  #   env_threads.append(t)

  # env = build_env(config)
  # env_thread = StoppableThread(step_env_with_add, replay_train, env, 0, name=f"env-step", start=True)
  train_env = build_env(config, seed=0, mode='train', train_ratio=0.9)
  eval_env = build_env(config, seed=0, mode='eval', train_ratio=0.9)

  # Checkpoint processing
  if config.run.save_every > 0:
    should_save = when.Clock(config.run.save_every)
    checkpoint = Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    if config.run.from_checkpoint:
      checkpoint.load(config.run.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we just saved.

  # try:
  while step < config.run.steps:

    # ==================== Step env, load data, infer, add state in
    if not train_step_stop:
      train_data = train_env.step()
      train_data = {k: v for k, v in train_data.items() if not k.startswith('log/')}
      prep_train_data = preprocess_observation(train_data)
      train_obs = {k: v for k, v in prep_train_data.items() if k in train_env.obs_space}

      # infering for evaluation
      logs = {k: v for k, v in train_data.items() if k.startswith('log/')}
      carry_infer_train, preds, outs = agent.infer(carry_infer_train, train_obs)
      # infer_fps.step(1)
      # outs = {k: np.zeros(v.shape, dtype=v.dtype)[None] for k, v in agent.ext_space.items() if k not in ('step_id', 'consec')}
      prep_outs = post_process_outs(outs)
      train_trans = {**train_data, **prep_outs, **logs}
      # print(f"[run] trans: {trans.keys()}")

      # Log step
      logfn(train_trans)

      # Add to replay buffers
      replay_train.add(train_trans, 0)

      # Finally, set train step to stop condition if satisfy certain condition
      if train_obs['is_dataset_last']:
        train_step_stop = True

    #======== [For evaluation] Step env, load data, infer, add state in
    if not eval_step_stop:
      # With this, we don't really take advantage of the saved state (we don't infer the state anymore)
      eval_data = eval_env.step()
      eval_data = {k: v for k, v in eval_data.items() if not k.startswith('log/')}
      prep_eval_data = preprocess_observation(eval_data)
      eval_obs = {k: v for k, v in prep_eval_data.items() if k in eval_env.obs_space}

      # infering for evaluation
      logs = {k: v for k, v in eval_data.items() if k.startswith('log/')}
      carry_infer_eval, preds, outs = agent.infer(carry_infer_eval, eval_obs)
      # infer_fps.step(1)
      # outs = {k: np.zeros(v.shape, dtype=v.dtype)[None] for k, v in agent.ext_space.items() if k not in ('step_id', 'consec')}
      prep_outs = post_process_outs(outs)
      eval_trans = {**eval_data, **prep_outs, **logs}

      # Add to replay buffers
      replay_eval.add(eval_trans, 0)

      # Set eval step to stop condition if satisfy certain condition
      if len(replay_eval) >= 100 * config.batch_size * (config.report_length + config.replay_context) * config.consec_report:
        eval_step_stop = True
        print(f"[run] eval_step_stop set to True, replay_eval length: {len(replay_eval)}", color='yellow')


    #====================== training
    trained = trainfn(None, None)

    # If report is needed, report the agent
    needed = config.batch_size * (config.report_length + config.replay_context) * config.consec_report
    if should_report(step) and len(replay_eval) >= needed:
      reportfn(None, None)

    # Logging and saving
    if int(step) % 10 == 0: # This save quite some time
      # Logging things
      if should_log(step):
        logger.add(train_agg.result())
        logger.add(val_agg.result())
        logger.add(usage.stats(), prefix='usage')
        # logger.add({'fps/infer': infer_fps.result()})
        logger.add({'fps/train': train_fps.result()})
        logger.add(epstats.result(), prefix='epstats')
        logger.add({'timer': common.timer.stats()['summary']})
        logger.write()
        # print(f"Replay size: {len(replay_train)}", color='green')

      if should_save(step):
        checkpoint.save()

    # Finally, increment the step()
    if trained:
      step.increment()

  # except Exception as e:
  #   raise e

  # finally:
  #   # for t in env_threads:
  #   #   t.stop(wait=True)
  #   env_thread.stop()
  # Finally, save the model
  checkpoint.save()


if __name__ == '__main__':
  if check_vscode_interactive():
    _args = [
      "--logroot=logs",
      "--expname=0",
    ]
  else:
    _args = sys.argv[1:]
  config = load_config_from_yaml(pathlib.Path(__file__).parent.parent / 'eet' / 'configs.yaml', _args)
  logdir = pathlib.Path(config.logdir)
  run(config)




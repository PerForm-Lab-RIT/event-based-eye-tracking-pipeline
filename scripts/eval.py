"""
File: eval.py
Author: Viet Nguyen
Date: 2025-05-28

Description: This script is used to evaluate the event-based eye tracking models.
Currently, this file only measure infer fps
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
from lib import Counter

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
  eval_step = Counter(0)
  logger.step = eval_step # Use eval_step for evaluation
  usage = common.Usage(**config.run.usage)
  val_agg = common.Agg()
  infer_fps = common.FPS()
  episode_stats = common.Agg() # temporary stats per episode
  epstats = common.Agg() # the final episode stats to be added to the logger

  should_log = when.Clock(config.run.log_every)
  should_report = when.Clock(config.run.report_every)

  agent: Agent = build_agent(config)
  carry_infer = agent.init_infer(1)
  # report_carry = agent.init_report(config.batch_size)
  # replay_eval: Replay = build_replay(config, "replay", mode="eval")
  # dataset_eval = iter(agent.stream(build_stream(config, replay_eval, "report")))

  def logfn(tran: Dict[str, np.ndarray]):
    # tran: (...), no batch, no seq length
    tran['is_dataset_first'] and episode_stats.reset()
    episode_stats.add(f'length', 1, agg='sum')

    if tran['is_dataset_last']:
      # print("[logfn] episode end", color='yellow')
      result = episode_stats.result()
      logger.add({
          f'length': result.pop(f'length'),
      }, prefix='eval_episode')
      epstats.add(result)

  def preprocess_observation(obs: Dict[str, np.ndarray]):
    proc = extend(obs)
    # print(jax.tree.map(lambda x: x.shape, proc), color='blue')
    # return nn.cast(proc)
    return proc

  def post_process_outs(outs: Dict[str, np.ndarray]):
    # return tree.map(lambda x: x[0], outs)
    return takefirst(outs)

  env = build_env(config, seed=0, mode="eval", train_ratio=0.9)

  # Checkpoint processing
  if config.run.save_every > 0:
    should_save = when.Clock(config.run.save_every)
    checkpoint = Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    if config.run.from_checkpoint:
      checkpoint.load(config.run.from_checkpoint)
    checkpoint.load()
    should_save(step)  # Register that we just saved.


  # Test for one episode only
  while True:

    # Step env, load data, infer, add state in
    data = env.step()
    data = {k: v for k, v in data.items() if not k.startswith('log/')}
    prep_data = preprocess_observation(data)
    obs = {k: v for k, v in prep_data.items() if k in env.obs_space}

    # infering for evaluation
    logs = {k: v for k, v in data.items() if k.startswith('log/')}
    carry_infer, preds, outs = agent.infer(carry_infer, obs)
    infer_fps.step(1)
    prep_outs = post_process_outs(outs)
    trans = {**data, **prep_outs, **logs}

    # Log step
    logfn(trans)

    # Add to replay buffers
    # replay_eval.add(trans, 0)

    # If report is needed, report the agent
    # needed = config.batch_size * (config.report_length + config.replay_context) * config.consec_report
    # if should_report(step) and len(replay_eval) >= needed:
    #   report_agg = common.Agg()
    #   for _ in range(config.run.report_batches):
    #     batch = next(dataset_eval)
    #     report_carry, mets = agent.report(report_carry, batch)
    #     report_agg.add(mets)
    #   logger.add(report_agg.result(), prefix='report')

    # Logging and saving
    if int(eval_step) % 10 == 0 or trans['is_dataset_last']: # This save quite some time
      # Logging things
      if should_log(eval_step) or trans['is_dataset_last']:
        logger.add(val_agg.result())
        logger.add(usage.stats(), prefix='usage')
        logger.add({'fps/infer': infer_fps.result()})
        logger.add(epstats.result(), prefix='eval_epstats')
        logger.add({'eval_timer': common.timer.stats()['summary']})
        logger.write()
        # print(f"Replay size: {len(replay_train)}", color='green')

    eval_step.increment()
    if trans['is_dataset_last']:
      break


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




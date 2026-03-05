# %%

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))
from typing import List, Dict, Tuple
import os
import json
import subprocess

# LOG_ROOT_DIR = pathlib.Path("~/tsxai/ai-pipeline/experimental/eet/logs_official/").expanduser()
# LOG_ROOT_DIR = pathlib.Path("/shared/rc/eyeseg/eet/logs/")
LOG_ROOT_DIR = pathlib.Path("logs").expanduser()
os.makedirs(LOG_ROOT_DIR, exist_ok=True)


INITIAL_TRAIN_SCRIPT = "#!/bin/bash\npython scripts/train.py "
INITIAL_EVAL_SCRIPT = "#!/bin/bash\npython scripts/eval.py "

DEPRECATED_CUSTOM2_PARAMS_ADDON = "--dynamics.ais2m.imglayers 2 --dynamics.ais2m.obslayers 2 --dynamics.ais2m.dynlayers 1 --dynamics.ais2m.alphalayers 2 --dynamics.ais2m.deter 32 --dynamics.ais2m.stoch 16 "


PARAM_LIST = [
  '500k',
  # '2m',
  # '10m'
]

ENV_LIST = [
  '3et',
  # 'eveye'
]

# 6 agents
agent_list = [
  # 'custom2',
  'cnn',
  # 'cnngru',
  # 'cnnbigru',
  # 'cbconvlstm',
  # 'mamba',
  # 'mambapupil'
]

IMAGE_SIZES = [
  # '80x60',
  '160x120'
]

BATCH_SIZE_MAPPING = {
  '80x60': {
    '500k': 128,
    '2m': 64,
    '10m': 32
  },
  '160x120': {
    '500k': 64,
    '2m': 32,
    '10m': 16
  },
}

USE_REPLAY_CONTEXT = {
  'custom2': True, # exclusive to this agent
  'cnn': False,
  'cnngru': False,
  'cnnbigru': False,
  'cbconvlstm': False,
  'mamba': False,
  'mambapupil': False
}


STEP_MAPPING = {
  '3et': 500_000,
  'eveye': 500_000,
}


EVENT_REPRS = [
  'binary',
  'binarep',
  'histogram',
  'voxelgrid',
  'eventframe'
]



def compose_executables(agent: str, env: str, image_size: str,
      params: str, trial: int, event_repr: str = None) -> Tuple[str, str]:
  """
  This function composes the executable commands for the given agent, parameter, and environment.
  It returns a list of commands, each corresponding to a different trial.
  Example: python

  Args:
    agent (str): The agent to use.
    param (str): The parameter to use.
    env (str): The environment to use.
  """

  output = ""

  # Check expname
  expdirname = f"{agent}_{params}-{env}_{image_size}"
  if event_repr is not None:
    expdirname += f"-{event_repr}"
  main_expdir = pathlib.Path(LOG_ROOT_DIR) / expdirname
  main_expdir.mkdir(exist_ok=True)
  expname = f"{expdirname}/trial_{trial}"
  main_expdir = pathlib.Path(LOG_ROOT_DIR) / expname
  main_expdir.mkdir(exist_ok=True)

  # NOTE: comment this out if running normally
  # if not (main_expdir / "checkpoint.ckpt").exists():
  #   return None, None

  # Main argument building
  kwargs_str = ""
  env_primitive = f"{env}_{image_size}"
  kwargs_str += f"--configs {env_primitive} {agent}_{params} "
  if event_repr is not None:
    kwargs_str += f"et_{event_repr} "
  kwargs_str += f"--logroot {LOG_ROOT_DIR} "
  kwargs_str += f"--expname {expname} "
  kwargs_str += f"--agent {agent} "
  kwargs_str += f"--batch_size {BATCH_SIZE_MAPPING[image_size][params]} "
  kwargs_str += f"--seed {trial} "
  kwargs_str += f"--run.steps {STEP_MAPPING[env]} "

  # Comment this out to use new model
  if agent == 'custom2':
    kwargs_str += DEPRECATED_CUSTOM2_PARAMS_ADDON

  # Replay context settings
  if USE_REPLAY_CONTEXT[agent]:
    replay_context = 1 # number of steps to look back for replay
    consec_train = 1 # number of batches to train per step
    consec_report = 1
  else:
    replay_context = 1
    consec_train = 1
    consec_report = 1
  kwargs_str += f"--replay_context {replay_context} "
  kwargs_str += f"--consec_train {consec_train} "
  kwargs_str += f"--consec_report {consec_report} "

  # Build training script
  # NOTE: Currently, the train kwargs are currently being commented out
  train_kwargs_str = kwargs_str
  # train_kwargs_str += f"--logger.outputs jsonl " # don't log tensorboard, it blows up the storage. Tensorbord file is large.
  output += INITIAL_TRAIN_SCRIPT + train_kwargs_str + "\n"

  # Build evaluation script
  output += INITIAL_EVAL_SCRIPT + kwargs_str + " --run.log_every=5 " + "\n\n"

  if output == "":
    return None, None
  return output, expname



### MAIN SCRIPT ###
# NOTE: I have changed to the next trial and update the submission script to optimize
# Basically, time limit is 1 day, then we can run continuously
# we only use 100gb of cpu memory, let's hope it does not crash
# lower time is better
# 0-> 9: RC
# 10 -> 19: Personal
# 20 -> 29: PL4
for trial in range(0, 1):
  for image_size in IMAGE_SIZES:
    for params in PARAM_LIST:
      for env in ENV_LIST:
        for agent in agent_list:
          # for event_repr in EVENT_REPRS:
            # Compose the bash script
            # bashstr, expname = compose_executables(agent, env, image_size, params, trial, event_repr)
            bashstr, expname = compose_executables(agent, env, image_size, params, trial)
            print(f"get here, {bashstr}")
            if bashstr is not None:
              print(bashstr)
              # Write the command to a file and submit a job
              with open("command.lock", "w") as f:
                f.write(bashstr)
              subprocess.call("bash command.lock", shell=True)
              if os.path.exists("command.lock"):
                os.remove("command.lock")



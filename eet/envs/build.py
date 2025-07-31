"""
File: build.py
Author: Viet Nguyen
Date: 2025-06-12

Description: Build the environment
"""

import importlib
import pathlib
import os

def build_env(config, **overrides):
  # NOTE: task should be a list, but for now, we only have one task
  task = config.task
  ctor = {
    'threeet': 'eet.envs.threeet:ThreeETEnv',
    'eveye': 'eet.envs.eveye:EVEyeEnv',
  }[task]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(task, {})
  kwargs.update(overrides)
  env = ctor(**kwargs)
  return env



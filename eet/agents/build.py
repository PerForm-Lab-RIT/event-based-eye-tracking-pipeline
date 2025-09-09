"""
File: build.py
Author: Viet Nguyen
Date: 2025-05-30

Description: Build agent utilities
"""

import sys
from lib.common import print
from lib.common import Config
from eet.envs.build import build_env

from .dummy import DummyAgent
from .cnn import CNNAgent
from .cnngru import CNNGRUAgent
from .cnnbigru import CNNBidirectionalGRUAgent
from .cbconvlstm import CBConvLSTMAgent
from .mamba import MambaAgent
from .mambapupil import MambaPupilAgent
from .custom import CustomAgent
from .custom2 import Custom2Agent

def build_agent(config: Config):
  # return RAMAgent(config, name='ram')
  env = build_env(config)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  label_space = {k: v for k, v in env.label_space.items() if notlog(k)}
  env.close()
  return {
    'dummy': DummyAgent,
    'cnn': CNNAgent,
    'cnngru': CNNGRUAgent,
    'cnnbigru': CNNBidirectionalGRUAgent,
    'cbconvlstm': CBConvLSTMAgent,
    'mamba': MambaAgent,
    'mambapupil': MambaPupilAgent,
    'custom': CustomAgent,
    'custom2': Custom2Agent,
  }[config.agent](obs_space, label_space, config=config)


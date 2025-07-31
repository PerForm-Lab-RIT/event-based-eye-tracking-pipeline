"""
File: dummy.py
Author: Viet Nguyen
Date: 2025-05-30

Description: Dummy agent
"""

from typing import Dict, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
import collections
from functools import partial as bind

from lib import nn
from lib.agent.inactive import JAXAgent
from lib.common import Space


# NOTE: Working
class DummyAgent(JAXAgent):

  def __init__(self, obs_space, label_space, config):
    self.obs_space = obs_space
    self.label_space = label_space
    self.linear = nn.Linear(1, name='linear')
    self.opt = nn.Optimizer([self.linear], name='optimizer')

  @property
  def policy_keys(self):
    # return '^(enc|dyn|dec|pol)/'
    return '.*'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = Space(np.int32)
    spaces['stepid'] = Space(np.uint8, 20)
    return spaces

  def init_infer(self, batch_size):
    return ()

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()

  def infer(self, carry, obs):
    return carry, {}, {}

  def _loss(self, carry, data: Dict[str, jax.Array]):
    loss = ((self.linear(nn.cast(jnp.zeros((4, 2)))).astype(jnp.float32) - 1)**2).mean()
    return loss, {}

  def train(self, carry, data):
    opt_mets, mets = self.opt(self._loss, carry, data, has_aux=True)
    mets.update(opt_mets)
    return carry, {}, mets

  def report(self, carry, data):
    return carry, {}, {}



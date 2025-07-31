"""
File: base.py
Author: Viet Nguyen
Date: 2025-03-12

Description: Base class for all environments
"""

from typing import Dict, List
import numpy as np

class InactiveEnv:

  def __repr__(self):
    return (
        f'{self.__class__.__name__}('
        f'obs_space={self.obs_space}'
        f', label_space={self.label_space})')

  @property
  def obs_space(self):
    # The observation space must contain the keys is_first (first in the sequence),
    #   is_last (last in the sequence), is_dataset_first (normally, we don't need this),
    #   is_dataset_last (end of the whole dataset).
    #   By convention, keys starting with 'log/' are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')

  @property
  def label_space(self):
    # The label space contain the output space of the model or the data field that we are predicting
    raise NotImplementedError('Returns: dict of spaces')

  def step(self):
    raise NotImplementedError('Returns: dict. Return data that combine both obs_space and label_space')

  def close(self):
    pass

  def render(self) -> Dict[str, np.ndarray] | np.ndarray | List[np.ndarray]:
    raise NotImplementedError('Returns: dict of images or image or list of images')


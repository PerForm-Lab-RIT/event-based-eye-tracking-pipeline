"""
File: common.py
Author: Viet Nguyen
Date: 2025-02-13

Description: Implement common utilities for the project
"""

import os, sys, pathlib
import math
import numpy as np
from datetime import datetime
import sys
from typing import Dict, Any
from PIL import Image
import io
try:
  import torch
except ImportError:
  pass

from ..common import tree

COMMON_NP_CONVERSION = {
  np.floating: np.float32,
  np.signedinteger: np.int64,
  np.uint8: np.uint8,
  bool: bool,
}

def get_root_dir() -> pathlib.Path:
  return pathlib.Path(__file__).parent.parent.parent.absolute()

def get_src_dir() -> pathlib.Path:
  return pathlib.Path(__file__).parent.parent.absolute()

def timestamp(now=None, millis=False):
  now = datetime.now() if now is None else now
  string = now.strftime("%Y%m%dT%H%M%S")
  if millis:
    string += f'F{now.microsecond:06d}'
  return string

def round_half_up(n):
  if n - int(n) >= 0.5:
    return math.ceil(n)
  return round(n)

def check_vscode_interactive() -> bool:
  if hasattr(sys, 'ps1'):
    return True # ipython on Windows or WSL
  else: # check on linux: https://stackoverflow.com/a/39662359
    try:
      shell = get_ipython().__class__.__name__
      if shell == 'ZMQInteractiveShell':
        return True   # Jupyter notebook or qtconsole
      elif shell == 'TerminalInteractiveShell':
        return False  # Terminal running IPython
      else:
        return False  # Other type (?)
    except NameError:
      return False      # Probably standard Python interpreter


def tensorstats(tensor):
  return {
    'mean': tensor.mean(),
    'std': tensor.std(),
    'mag': np.abs(tensor).max(),
    'min': tensor.min(),
    'max': tensor.max()
  }


def subsample(values, amount=512):
  values = values.flatten()
  if len(values) > amount:
    values = np.random.permutation(values)[:amount]
  return values


def convert(value):
  value = np.asarray(value)
  if value.dtype not in COMMON_NP_CONVERSION.values():
    for src, dst in COMMON_NP_CONVERSION.items():
      if np.issubdtype(value.dtype, src):
        if value.dtype != dst:
          value = value.astype(dst)
        break
    else:
      raise TypeError(f"Object '{value}' has unsupported dtype: {value.dtype}")
  return value


def convert_pytorch_to_numpy(node: Dict[str, Any]) -> Dict[str, Any]:
  """Convert PyTorch tree to numpy arrays for JAX.

  Args:
      node (Dict[str, Any]): _description_

  Returns:
      Dict[str, Any]: _description_
  """
  fn = lambda x: x.numpy() if torch.is_tensor(x) else x
  return tree.map(fn, node)


def plt_figure_to_numpy(figure):
  """Converts a Matplotlib figure to a NumPy array."""
  buf = io.BytesIO()
  figure.savefig(buf, format='png')
  buf.seek(0)
  image = Image.open(buf)
  return np.array(image)


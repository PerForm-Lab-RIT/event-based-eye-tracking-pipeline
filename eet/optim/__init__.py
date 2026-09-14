"""Optimization utilities.
- LaProp optimizer (see optim/laprop.py for upstream license header)
- Adaptive Gradient Clipping (AGC)
"""

from .agc import clip_grad_agc_
from .laprop import LaProp
from .opt import OptimizerEngine
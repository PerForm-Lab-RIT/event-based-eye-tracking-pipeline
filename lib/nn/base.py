"""
File: base.py
Author: Viet Nguyen
Date: 2024-01-23

Description: Adapted from Danijar Hafner's code
"""

from typing import Callable, Tuple, List, Dict
import math
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import jax.ad_checkpoint as adc

from . import utils as jaxutils
from . import functional as F
from . import ninjax as nj
from . import utils
from . import internal

f32 = jnp.float32
i32 = jnp.int32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
castpd = jaxutils.cast_to_param

def get_act(name):
  if callable(name):
    return name
  elif name == 'none' or name == 'identity' or name == None:
    return lambda x: x
  elif name == 'relu':
    # JAX's relu does not have gradient at 0: https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L54-L88
    # f'(x) can have gradient of 1 at 0. https://stackoverflow.com/a/76396054/14861798
    @jax.custom_jvp
    @jax.jit
    def custom_relu(x):
      return jnp.maximum(x, 0)
    custom_relu.defjvps(lambda g, ans, x: jax.lax.select(x >= 0, g, jax.lax.full_like(g, 0)))
    return custom_relu
  elif name == 'gelu_tanh':
    return F.gelu_tanh
  elif name == 'gelu_quick':
    return lambda x: x * jax.nn.sigmoid(1.702 * x)
  elif name == 'relu2':
    return lambda x: jnp.square(jax.nn.relu(x))
  elif name == 'swiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      return jax.nn.silu(x) * y
    return fn
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)


def get_initializer(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return Initializer(dist, fan, 1.0)


class Initializer:

  def __init__(self, dist='trunc_normal', fan='in', scale=1.0):
    self.dist = dist
    self.fan = fan
    self.scale = scale

  def __call__(self, shape, dtype=jnp.float32, fshape=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    assert all(isinstance(x, int) for x in shape), (
      shape, [type(x) for x in shape])
    assert all(x > 0 for x in shape), shape
    fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
    fan = {
      'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
    }[self.fan]
    if self.dist == 'zeros':
      x = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      x = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      x = jax.random.normal(nj.seed(), shape)
      x *= np.sqrt(1 / fan)
    elif self.dist == 'trunc_normal':
      x = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      x *= 1.1368 * np.sqrt(1 / fan)
    elif self.dist == 'normed':
      x = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      x *= (1 / jnp.linalg.norm(x.reshape((-1, shape[-1])), 2, 0))
    else:
      raise NotImplementedError(self.dist)
    x *= self.scale
    x = x.astype(dtype)
    return x

  def __repr__(self):
    return f'Initializer({self.dist}, {self.fan}, {self.scale})'

  def __eq__(self, other):
    attributes = ('dist', 'fan', 'scale')
    return all(getattr(self, k) == getattr(other, k) for k in attributes)

  @staticmethod
  def compute_fans(shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = math.prod(shape[:-2])
      return (shape[-2] * space, shape[-1] * space)


class Linear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units: Tuple[int] | int):
    self.units = (units,) if isinstance(units, int) else tuple(units)

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    size = math.prod(self.units)
    shape = (x.shape[-1], size)
    x = x @ self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), size).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], *self.units))
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale


class BlockLinear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units, blocks):
    """
    Initialize BlockLinear layer.

    Args:
      units (int): Total number of output units
      blocks (int): Number of blocks to split input and output
    """
    assert isinstance(units, int), (units, type(units))
    assert blocks <= units and units % blocks == 0, (blocks, units)
    self.units = units
    self.blocks = blocks

  def __call__(self, x):
    """
    Apply block linear transformation.

    Args:
      x (jax.Array): Input tensor with shape (..., input_features)
        where input_features must be divisible by self.blocks

    Returns:
      jax.Array: Output tensor with shape (..., self.units)
    """
    jaxutils.ensure_dtypes(x)
    assert x.shape[-1] % self.blocks == 0, (x.shape, self.blocks)
    insize = x.shape[-1]
    shape = (self.blocks, insize // self.blocks, self.units // self.blocks)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], self.blocks, insize // self.blocks))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*x.shape[:-2], self.units))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.units).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale


class Conv1D(nj.Module):
  # NOTE: TODO: Conv1D with groups have buggy gradient, debugging it

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) if isinstance(kernel, int) else tuple(kernel)
    self.stride = stride

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    # x: (..., length, in_channels)
    # kernel: (kernel, in_channels // groups, out_channels)
    shape = (self.kernel[0], x.shape[-1] // self.groups, self.depth)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.transp:
      assert self.pad == 'same', self.pad
      # Manual implementation of fractionally strided convolution (rarely used for 1D)
      x = x.repeat(self.stride, -2)
      mask = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)
      x *= mask[:, None]
      stride = (1,)
    else:
      stride = (self.stride,)
    # NHWC for 1D: treat as NLC (batch, length, channels)
    x = jax.lax.conv_general_dilated(
        x, kernel, stride, self.pad.upper(),
        feature_group_count=self.groups,
        dimension_numbers=('NLC', 'LIO', 'NLC'))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.depth).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale


class Conv2D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 2 if isinstance(kernel, int) else kernel
    self.stride = stride

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.transp:
      assert self.pad == 'same', self.pad
      # Manual implementation of fractionally strided convolution because the
      # cuDNN implementation used by XLA has bugs and performance issues.
      x = x.repeat(self.stride, -2).repeat(self.stride, -3)
      maskh = ((jnp.arange(x.shape[-3]) - 1) % self.stride == 0)[:, None]
      maskw = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)[None, :]
      x *= (maskh * maskw)[:, :, None]
      stride = (1, 1)
    else:
      stride = (self.stride, self.stride)
    x = jax.lax.conv_general_dilated(
        x, kernel, stride, self.pad.upper(),
        feature_group_count=self.groups,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.depth).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs) * self.outscale


class Conv3D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 3 if isinstance(kernel, int) else kernel
    self.stride = (stride,) * 3 if isinstance(stride, int) else stride

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (*self.kernel, x.shape[-1], self.depth)
      kernel = self.value('kernel', get_initializer(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_transpose(
          x, kernel, self.stride, self.pad.upper(),
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    else:
      shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
      kernel = self.value('kernel', get_initializer(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_general_dilated(
          x, kernel, self.stride, self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    if self.bias:
      x += self.value('bias', get_initializer(self.binit), self.depth).astype(x.dtype)
    return x


class Norm(nj.Module):

  axis: tuple = (-1,)
  eps: float = 1e-4
  scale: bool = True
  shift: bool = True

  def __init__(self, impl):
    if '1em' in impl:
      impl, exp = impl.split('1em')
      self._fields['eps'] = 10 ** -int(exp)
    self.impl = impl

  def __call__(self, x):
    jaxutils.ensure_dtypes(x)
    dtype = x.dtype
    x = f32(x)
    axis = [a % x.ndim for a in self.axis]
    shape = [x.shape[i] if i in axis else 1 for i in range(min(axis), x.ndim)]
    if self.impl == 'none':
      pass
    elif self.impl == 'rms':
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      scale = self._scale(shape, x.dtype)
      x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
    elif self.impl == 'layer':
      mean = x.mean(axis, keepdims=True)
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = adc.checkpoint_name(var, 'small')
      scale = self._scale(shape, x.dtype)
      shift = self._shift(shape, x.dtype)
      x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    else:
      raise NotImplementedError(self.impl)
    x = x.astype(dtype)
    return x

  def _scale(self, shape, dtype):
    if not self.scale:
      return jnp.ones(shape, dtype)
    return self.value('scale', jnp.ones, shape, f32).astype(dtype)

  def _shift(self, shape, dtype):
    if not self.shift:
      return jnp.zeros(shape, dtype)
    return self.value('shift', jnp.zeros, shape, f32).astype(dtype)


class MLP(nj.Module):

  act: str = 'silu'
  norm: str = 'rms'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, layers: int = 5, units: int = 1024):
    self.layers = layers
    self.units = units
    self.kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    self.outkw = dict(outscale=self.outscale, **self.kw)

  def __call__(self, x):
    shape = x.shape[:-1]
    x = x.astype(utils.COMPUTE_DTYPE)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self.layers):
      kw = self.kw if i < self.layers - 1 else self.outkw
      x = self.sub(f'linear{i}', Linear, self.units, **kw)(x)
      x = self.sub(f'norm{i}', Norm, self.norm)(x)
      x = get_act(self.act)(x)
    x = x.reshape((*shape, x.shape[-1]))
    return x


class GRU(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int):
    self.units = units

  def initial(self, batch_size):
    return jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE)

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): (B, U)
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert carry.dtype == utils.COMPUTE_DTYPE, carry.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    # print(f"[GRU] carry: {carry.shape}, inputs: {inputs.shape}, resets: {resets.shape}")
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = F.mask(carry, ~reset)
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('norm', Norm, self.norm)(x)
    x = self.sub('linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output


class BidirectionalGRU(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int):
    self.units = units

  def initial(self, batch_size):
    # Return initial states for both forward and backward GRUs
    forward_carry = jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE)
    backward_carry = jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE)
    return jnp.concatenate([forward_carry, backward_carry], axis=-1)

  def __call__(self, carry, inputs, resets):
    """Bidirectional GRU forward pass

    Args:
        carry (tuple): (forward_carry, backward_carry) each (B, U)
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)
        single (bool, optional): single step input. Defaults to False.

    Returns:
        tuple: (carry, outputs) where outputs is concatenated forward and backward outputs
    """
    forward_carry, backward_carry = jnp.split(carry, 2, axis=-1)
    assert forward_carry.dtype == utils.COMPUTE_DTYPE, forward_carry.dtype
    assert backward_carry.dtype == utils.COMPUTE_DTYPE, backward_carry.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype

    # Forward pass
    forward_carry, forward_outputs = nj.scan(
      lambda carry, args: self.forward_step(carry, *args),
      forward_carry, (inputs, resets), axis=1)

    # Backward pass - reverse the sequence
    reversed_inputs = jnp.flip(inputs, axis=1)
    reversed_resets = jnp.flip(resets, axis=1)

    backward_carry, backward_outputs = nj.scan(
      lambda carry, args: self.backward_step(carry, *args),
      backward_carry, (reversed_inputs, reversed_resets), axis=1)

    # Reverse the backward outputs to match original sequence order
    backward_outputs = jnp.flip(backward_outputs, axis=1)

    # Concatenate forward and backward outputs
    outputs = jnp.concatenate([forward_outputs, backward_outputs], axis=-1)

    return jnp.concatenate([forward_carry, backward_carry], axis=-1), outputs

  def forward_step(self, carry, inp, reset):
    # Forward GRU step
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = F.mask(carry, ~reset)
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('forward_norm', Norm, self.norm)(x)
    x = self.sub('forward_linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output

  def backward_step(self, carry, inp, reset):
    # Backward GRU step
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = F.mask(carry, ~reset)
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('backward_norm', Norm, self.norm)(x)
    x = self.sub('backward_linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output


# Operate similar to GRU
class MambaBlock(nj.Module):

  kernel: int = 4
  expand: int = 16 # Expansion factor for input projection => to compute d_inner
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, deter: int, units: int):
    self.units = units # units here is the d_model => or dimension of the input and output vectors of the Mamba block
    self.deter = deter # units here is the d_state => or the dimension of of the state space model's state vector
    self.inners = units * self.expand

  def initial(self, batch_size):
    return jnp.zeros((batch_size, self.inners, self.deter), utils.COMPUTE_DTYPE)

  def _scaled_winit(self, *args, **kwargs):
    return get_initializer(self.winit)(*args, **kwargs)

  def _A_winit(self):
    w = -jnp.arange(1, self.deter + 1)[None].repeat(self.inners, axis=0) # (16D, Z)
    w = jnp.float32(w)
    return w

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
      carry (jax.Array): (B, U)
      inputs (jax.Array): (B, T, I) or (B, I) if single
      resets (jax.Array): (B, T) or (B,) if single
      single (bool, optional): single step input. Defaults to False.

    Returns:
      carry (jax.Array): (B, U)
      outputs (jax.Array): (B, T, U) or (B, U) if single
    """
    ######### Input checks
    assert carry.dtype == utils.COMPUTE_DTYPE, carry.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    # Ensure the inputs are initialized correctly
    x = inputs[:, None] if single else inputs
    assert x.ndim == 3, f"Expected inputs to be 3D, got {x.ndim}D"
    _, seqlen, indims = x.shape

    ########## additional weights initialization needed
    A = self.value('A', self._A_winit).astype(utils.COMPUTE_DTYPE) # (16D, Z)
    D = self.value('D', jnp.ones, (self.inners,)).astype(utils.COMPUTE_DTYPE) # (16D,)

    ########## Main ops
    """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

    Args:
        x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

    Returns:
        output: shape (b, l, d)

    Official Implementation:
        class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
        mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

    """

    # project the inputs to the d_inner * 2 dimension, one used in main, and one used in res
    x = self.sub('inproj', Linear, self.inners * 2, bias=False, winit=self.winit)(x)
    x, res = jnp.split(x, 2, axis=-1) # split into two parts # (B, L, 16D), (B, L, 16D)

    # Extract local feature using a Conv1D
    # x = self.sub('conv', Conv1D, self.inners,
    #   kernel=self.kernel, stride=1, groups=self.inners,
    #   pad='same', winit=self.winit, binit=self.binit)(x) # (B, L, 16D)
    # conv1D with groups have buggy gradient, debugging it
    x = self.sub('local', Linear, self.inners, winit=self.winit, binit=self.binit)(x) # (B, L, 16D)
    x = jax.nn.silu(x) # non-linearity (B, L, 16D)

    ################## SSM Block
    delta_B_C = self.sub('xproj', Linear,
      self.units + 2 * self.deter, bias=False, winit=self.winit)(x) # (B, L, D + 2Z)
    delta, B, C = jnp.split(
      delta_B_C,
      [self.units, self.units + self.deter],
      axis=-1
    ) # delta: (B, L, D). B, C: (B, L, Z)
    delta = self.sub('dtproj', Linear, self.inners, bias=False, winit=self.winit)(delta) # (B, L, 16D)
    delta = jax.nn.softplus(delta) # (B, L, 16D)

    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
        delta: shape (b, l, d_in)
        A: shape (d_in, n)
        B: shape (b, l, n)
        C: shape (b, l, n)
        D: shape (d_in,)

    Returns:
        output: shape (b, l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
    """
    # Selective scan starts here
    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    dA = jnp.exp(jnp.einsum('bld,dz->bldz', delta, A))
    dBu = jnp.einsum('bld,blz,bld->bldz', delta, B, x)

    # Selective scan
    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    # Note that the below is sequential, while the official implementation does a much faster parallel scan that
    # is additionally hardware-aware (like FlashAttention).
    # carry: (B, 16D, Z), carries: (B, L, 16D, Z), y: (B, L, 16D)
    carry, (carries, y) = nj.scan(
      lambda c, xs: self._step(c, *xs),
      carry, (dA, dBu, C), length=seqlen, axis=1)

    # Add skip connection
    y = y + x * D # (B, L, 16D)

    ############# End of SSM Block

    # Residual connection (again)
    y = y * jax.nn.silu(res)

    # Output projection
    outputs = self.sub('outproj', Linear, self.units, bias=False, winit=self.winit)(y) # (B, L, D)

    if single:
      outputs = outputs[:, 0]
      carries = carries[:, 0]

    return carry, carries, outputs

  def _step(self, carry, dA, dBu, C):
    """
    Single step of the selective scan.
    """
    carry = dA * carry + dBu
    outputs = jnp.einsum('bdz,bz->bd', carry, C)
    return carry, (carry, outputs)  # [batch, *state_dim], ( [batch, *state_dim], [batch, dim] )


class LSTM(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int):
    self.units = units

  def initial(self, batch_size):
    return (
      jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE), # cell state
      jnp.zeros((batch_size, self.units), utils.COMPUTE_DTYPE) # cell output
    )

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): ((B, U), (B, U))
        inputs (jax.Array): (B, T, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    c, h = carry
    assert c.dtype == utils.COMPUTE_DTYPE, c.dtype
    assert h.dtype == utils.COMPUTE_DTYPE, h.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

#   def step(self, carry, inp, reset):
#     # NOTE: When passing previous actions as input, ensure to zero out past
#     # actions on is_first and clip actions to bounds if needed.
#     kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
#     cell_state, cell_output = carry
#     cell_state = F.mask(cell_state, ~reset)
#     cell_output = F.mask(cell_output, ~reset)
#     _inp = jnp.concatenate([cell_output, inp], axis=-1)  # (B, I + U)
#     _inp = self.sub('norm', Norm, self.norm)(_inp)
#     # Forget in cell state
#     forget_logit = self.sub('forget', Linear, self.units, **kw)(_inp) # (B, U)
#     forget_probs = jax.nn.sigmoid(forget_logit) # (B, U)
#     cell_state = forget_probs * cell_state
#     # Input in cell state
#     input_proj = self.sub('input_proj', Linear, self.units, **kw)(_inp)  # (B, U)
#     input_proj = jax.nn.tanh(input_proj)  # (B, U)
#     input_logit = self.sub('input_logit', Linear, self.units, **kw)(_inp)  # (B, U)
#     input_probs = jax.nn.sigmoid(input_logit)  # (B, U)
#     cell_state += input_probs * input_proj
#     # Output the cell output (not the state)
#     out_logit = self.sub('output', Linear, self.units, **kw)(_inp)  # (B, U)
#     out_probs = jax.nn.sigmoid(out_logit)  # (B, U)
#     output = jax.nn.tanh(cell_state) * out_probs  # (B, U)
#     carry = (cell_state, output)
#     return carry, output

  # faster
  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    cell_state, cell_output = carry
    cell_state = F.mask(cell_state, ~reset)
    cell_output = F.mask(cell_output, ~reset)
    _inp = jnp.concatenate([cell_output, inp], axis=-1)  # (B, I + U)
    _inp = self.sub('norm', Norm, self.norm)(_inp)
    _inp_out = self.sub('main', Linear, 4 * self.units, **kw)(_inp)
    forget_logit, input_proj, input_logit, out_logit = jnp.split(_inp_out, 4, axis=-1)  # (B, U)
    # Forget in cell state
    forget_probs = jax.nn.sigmoid(forget_logit) # (B, U)
    cell_state = forget_probs * cell_state
    # Input in cell state
    input_proj = jax.nn.tanh(input_proj)  # (B, U)
    input_probs = jax.nn.sigmoid(input_logit)  # (B, U)
    cell_state += input_probs * input_proj
    # Output the cell output (not the state)
    out_probs = jax.nn.sigmoid(out_logit)  # (B, U)
    output = jax.nn.tanh(cell_state) * out_probs  # (B, U)
    carry = (cell_state, output)
    return carry, output


class ConvLSTM(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int, kernel: int | tuple):
    self.units = units
    self.kernel = kernel

  def initial(self, batch_size, image_size: tuple):
    width, height = image_size
    return (
      jnp.zeros((batch_size, height, width, self.units), utils.COMPUTE_DTYPE), # cell state
      jnp.zeros((batch_size, height, width, self.units), utils.COMPUTE_DTYPE) # cell output
    )

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): ((B, H, W, U), (B, H, W, U))
        inputs (jax.Array): (B, T, H, W, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    c, h = carry
    assert c.dtype == utils.COMPUTE_DTYPE, c.dtype
    assert h.dtype == utils.COMPUTE_DTYPE, h.dtype
    assert inputs.dtype == utils.COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    cell_state, cell_output = carry
    cell_state = F.mask(cell_state, ~reset)
    cell_output = F.mask(cell_output, ~reset)
    _inp = jnp.concatenate([cell_output, inp], axis=-1)  # (B, I + U)
    _inp = self.sub('norm', Norm, self.norm)(_inp)
    _inp_out = self.sub('main', Conv2D, depth=4 * self.units, kernel=self.kernel, stride=1, pad='same', **kw)(_inp)
    forget_logit, input_proj, input_logit, out_logit = jnp.split(_inp_out, 4, axis=-1)  # (B, U)
    # Forget in cell state
    forget_probs = jax.nn.sigmoid(forget_logit) # (B, U)
    cell_state = forget_probs * cell_state
    # Input in cell state
    input_proj = jax.nn.tanh(input_proj)  # (B, U)
    input_probs = jax.nn.sigmoid(input_logit)  # (B, U)
    cell_state += input_probs * input_proj
    # Output the cell output (not the state)
    out_probs = jax.nn.sigmoid(out_logit)  # (B, U)
    output = jax.nn.tanh(cell_state) * out_probs  # (B, U)
    carry = (cell_state, output)
    return carry, output


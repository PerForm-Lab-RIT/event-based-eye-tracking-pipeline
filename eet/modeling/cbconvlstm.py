from typing import Callable
from lib.nn import Initializer
import jax
import jax.numpy as jnp
import numpy as np

from lib.nn import utils
from lib.nn import ninjax as nj
from lib.nn import functional as F
from lib.nn.base import Norm, Conv2D


class CBConvLSTM(nj.Module):
  """CBConvLSTM module.
    A CBConvLSTM module that implements a change-based convolutional LSTM cell.
    https://arxiv.org/abs/2308.11771
  """

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def __init__(self, units: int, kernel: int | tuple):
    self.units = units
    self.kernel = kernel

  def macs(self, carry, inputs, resets) -> int:
    """Compute Multiply-Accumulate Operations (MACs) for a single forward pass.

    Args:

    Returns:
        int: The estimated number of MACs.
    """
    # Infer shapes. `inputs` can be a sequence (B, T, H, W, I) or a single step
    # (B, H, W, I). `carry` contains tensors shaped (B, H, W, U) and a
    # previous input with channels I which we can use to infer I when needed.
    try:
      ishape = inputs.shape
    except Exception:
      return 0
    if len(ishape) == 5:
      B, T, H, W, I = ishape
    elif len(ishape) == 4:
      B, H, W, I = ishape
      T = 1
    else:
      # unexpected shape
      return 0

    # Units and kernel
    U = int(self.units)
    if isinstance(self.kernel, tuple) or isinstance(self.kernel, list):
      k_h, k_w = self.kernel
    else:
      k_h = k_w = int(self.kernel)
    k_area = int(k_h) * int(k_w)

    # Channel sizes for conv: input channels = I + U (delta_inp concat delta)
    in_ch = I + U
    out_ch = 4 * U  # Conv2D produces 4*U channels then split into 4 gates

    # Per-step MACs
    # 1) Masking: three masks applied to tensors with U channels
    mask_macs_per_step = 3 * (B * H * W * U)

    # 2) Elementwise diffs and thresholds for delta (U) and delta_inp (I)
    diff_macs_per_step = B * H * W * (U + I)  # subtractions
    thresh_macs_per_step = B * H * W * (U + I)  # comparisons/select

    # 3) Normalization on combined tensor (B,H,W,I+U). Approximate as ~4 ops/elem
    combined_elems = B * H * W * (I + U)
    norm_macs_per_step = 4 * combined_elems

    # 4) Conv2D MACs: for each output element multiply-accumulate across
    # k_area * in_ch inputs. Count MACs = B*H*W*out_ch*k_area*in_ch
    conv_macs_per_step = B * H * W * out_ch * k_area * in_ch

    # 5) Activations: sigmoid/tanh for 4 gates (each U channels). Approx 4 ops each
    gate_elems = B * H * W * U
    activation_cost_per_elem = 4
    activations_macs_per_step = 4 * gate_elems * activation_cost_per_elem

    # 6) Elementwise gate math: multiplies/adds for gating and state updates
    # forget * state, input_prob * input_proj, add to state, output multiply
    gate_elemwise_macs_per_step = 4 * gate_elems
    # plus tanh(cell_state) before multiply with out_probs
    cell_tanh_macs_per_step = activation_cost_per_elem * gate_elems

    # Sum per-step
    per_step = (
      mask_macs_per_step +
      diff_macs_per_step +
      thresh_macs_per_step +
      norm_macs_per_step +
      conv_macs_per_step +
      activations_macs_per_step +
      gate_elemwise_macs_per_step +
      cell_tanh_macs_per_step
    )

    # Total across time steps
    total = int(per_step * int(T))
    return total


  def initial(self, batch_size: int, image_size: tuple, input_size: int):
    width, height = image_size
    return (
      jnp.zeros((batch_size, height, width, self.units), utils.COMPUTE_DTYPE), # cell state
      jnp.zeros((batch_size, height, width, self.units), utils.COMPUTE_DTYPE), # cell output
      jnp.zeros((batch_size, height, width, self.units), utils.COMPUTE_DTYPE), # previous cell output
      jnp.zeros((batch_size, height, width, input_size), utils.COMPUTE_DTYPE) # previous input
    )

  def __call__(self, carry, inputs, resets, single=False):
    """_summary_

    Args:
        carry (jax.Array): ((B, H, W, U), (B, H, W, U), (B, H, W, U))
        inputs (jax.Array): (B, T, H, W, I)
        resets (jax.Array): (B, T)
        single (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # print(f"[CBConvLSTM.__call__] carry: {carry}")
    c, h, hm1, prevlat = carry
    assert c.dtype == utils.COMPUTE_DTYPE, c.dtype
    assert h.dtype == utils.COMPUTE_DTYPE, h.dtype
    assert hm1.dtype == utils.COMPUTE_DTYPE, hm1.dtype
    assert prevlat.dtype == utils.COMPUTE_DTYPE, prevlat.dtype
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
    cell_state, cell_output, prev_cell_output, prevlatinp = carry
    cell_state = F.mask(cell_state, ~reset)
    cell_output = F.mask(cell_output, ~reset)
    prev_cell_output = F.mask(prev_cell_output, ~reset)

    # Delta encoder
    delta = cell_output - prev_cell_output  # (B, H, W, U)
    delta_inp = inp - prevlatinp  # (B, H, W, I)
    threshold = 0.002
    delta = jnp.where(delta < threshold, 0.0, delta)  # (B, H, W, U)
    delta_inp = jnp.where(delta_inp < threshold, 0.0, delta_inp)  # (B, H, W, I)
    combined = jnp.concatenate([delta_inp, delta], axis=-1)  # concatenate along channel axis

    # Convolutional LSTM cell
    _inp = self.sub('norm', Norm, self.norm)(combined)
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
    carry = (cell_state, output, cell_output, inp)
    # Return the output together with the current input as previous input of the next step
    return carry, output


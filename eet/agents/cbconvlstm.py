"""
File: cnn.py
Author: Viet Nguyen
Date: 2025-05-31

Description: develop CNN agent
"""

# import sys, pathlib
# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
# sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.parent))

from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import re

from lib.common import Space, print
from lib.agent.inactive import JAXAgent
from lib import nn
from lib.nn import functional as F

from modeling.encoder import CBConvLSTMEncoder
from .utils import *


class CBConvLSTMAgent(JAXAgent):

  def __init__(self, obs_space, label_space, config, *args, **kwargs):
    self.obs_space = obs_space
    self.label_space = label_space
    exclude = ('is_first', 'is_last', 'is_dataset_first', 'is_dataset_last')
    enc_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.encoder.prockeys, k))}
    self.encoder = CBConvLSTMEncoder(enc_space, **config.encoder.cbconvlstm, name="enc")
    _label_space = {k: Space(np.float32, v.shape, v.low, v.high) if ispupilcentroid(v) else v for k, v in label_space.items()}
    # print(f"[CNNAgent.__init__] _label_space: {_label_space}", color='green')
    self.head = nn.MLPHead(_label_space, {k: 'identity' for k in label_space}, **config.head.general, name='head')
    modules = [self.encoder, self.head]
    self.opt = nn.Optimizer(modules, **config.opt, name="opt")

    self.imgkeys = [k for k in obs_space if isimage(obs_space[k])]
    centkey = [k for k in label_space if ispupilcentroid(label_space[k])]
    assert len(centkey) <= 1
    if len(centkey) == 1:
      self.centkey = centkey[0]
    else:
      self.centkey = None

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = Space(np.int32)
    spaces['stepid'] = Space(np.uint8, 20)
    return spaces

  def init_infer(self, batch_size):
    example_obs = jax.tree.map(lambda x: jnp.zeros((batch_size, *x.shape)), self.obs_space)
    carry = self.encoder.initial(example_obs, single=True)
    return (carry,)

  def init_train(self, batch_size):
    return self.init_infer(batch_size)

  def init_report(self, batch_size):
    return self.init_infer(batch_size)

  def infer(self, carry, obs, mode="train"):
    (enc_carry,) = carry
    obs = preprocess_data(obs)
    feat, enc_carry = self.encoder(obs, obs['is_first'], enc_carry, training=False, single=True) # (B, dim)
    dist = self.head(feat, bdims=1)
    result = {k: jax.nn.tanh(v.pred()) for k, v in dist.items()}
    carry = (enc_carry,)
    return carry, result, {}

  def gflops(self, carry, obs):
    # Estimate MACs for the infer forward path used in `infer`.
    # infer does: feat, enc_carry = self.encoder(obs, obs['is_first'], enc_carry, training=False, single=True)
    #            dist = self.head(feat, bdims=1)

    total = 0
    # Use encoder.macs where available. The encoder.macs expects (obs, reset, carry)
    reset = obs.get('is_first', None)
    (enc_carry,) = carry
    enc_m = self.encoder.macs(obs, reset, enc_carry)
    total += int(enc_m)

    # head.macs expects input and bdims; infer calls head(feat, bdims=1)
    # Determine bdims from reset / data shape: infer() uses single=True (bdims=1),
    # but loss/train passes sequences (bdims=2). Support both.
    reset = obs.get('is_first', None)
    if reset is None:
      bdims = 1
    else:
      bdims = 1 if reset.ndim == 1 else 2

    # Try to obtain a sample feature by running the encoder in the matching mode.
    B = int(obs['is_first'].shape[0]) if 'is_first' in obs else 1
    if bdims == 1:
      sample_feat, _ = self.encoder(obs, obs['is_first'], enc_carry, training=False, single=True)
    else:
      # bdims == 2: encoder expects reset shape (B, T)
      sample_feat, _ = self.encoder(obs, obs['is_first'], enc_carry, training=False, single=False)

    head_m = self.head.macs(sample_feat, bdims)
    total += int(head_m)

    # final tanh preds: negligible but count one op per output element
    # estimate number of output scalars from head by checking label_space shapes
    out_elems = 0
    for k, space in self.label_space.items():
      if hasattr(space, 'shape') and space.shape:
        out_elems += int(np.prod(space.shape))
      else:
        out_elems += 1
    # If bdims==2, account for time dimension
    if 'reset' in locals() and reset is not None and getattr(reset, 'ndim', 1) == 2:
      T = int(reset.shape[1])
    else:
      T = 1
    total += int(B * T * out_elems * 4)  # approx 4 ops per tanh per example

    return macs2gflops(total)

  def _loss(self, carry, data, training=True):
    metrics = {}
    data = preprocess_data(data)
    reset = data['is_first']
    label_mask = data['label_mask'] # if the label is valid (B, T)

    (enc_carry,) = carry
    # print(f"[CBConvLSTMAgent._loss] enc_carry: {enc_carry}", color="red")
    feat, enc_carry = self.encoder(data, reset, enc_carry, training=training, single=False) # (B, dim)
    dists = self.head(feat, bdims=2)
    losses = {}
    for key, dist in dists.items():
      space, value = self.label_space[key], data[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = normpupilcentroid(nn.f32(value), space.high + 1) if (ispupilcentroid(space) or key == 'pupil') else value
      prediction = jax.nn.tanh(dist.pred())
      _loss = ((prediction - nn.sg(target))**2).sum(-1) # (B, T) # MSE Loss
      losses[key] = F.mask(_loss, label_mask) # (B, T) x (B, T)
    # Assert shape
    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Compute accuracy metrics
    for k, dist in dists.items():
      if ispupilcentroid(self.label_space[k]):
        pred_normalized = jnp.clip(dist.pred(), -1, 1)
        pred = unnormpupilcentroid(pred_normalized, self.label_space[k].high + 1)
        label = nn.f32(data[k])
        distance = jnp.linalg.norm(pred - label, axis=-1) # (B, T)
        # for some frame, the label might not be available, so we do this
        p5 = jnp.sum(F.mask(distance <= 5.0, label_mask)) / jnp.sum(label_mask) * 100
        p10 = jnp.sum(F.mask(distance <= 10.0, label_mask)) / jnp.sum(label_mask) * 100
        p15 = jnp.sum(F.mask(distance <= 15.0, label_mask)) / jnp.sum(label_mask) * 100
        p20 = jnp.sum(F.mask(distance <= 20.0, label_mask)) / jnp.sum(label_mask) * 100
        p50 = jnp.sum(F.mask(distance <= 50.0, label_mask)) / jnp.sum(label_mask) * 100
        metrics.update({f'{k}/p5': p5, f'{k}/p10': p10, f'{k}/p15': p15, f'{k}/p20': p20, f'{k}/p50': p50})

        # compute the metrics according to the original scale of the dataset, not the preprocessed one.
        # In this case, we will use a fixed height and width of 320x320
        unnorm2_pred = unnormpupilcentroid(pred_normalized, np.asarray([320, 320]))
        unnorm_label = label / (self.label_space[k].high + 1) * np.asarray([320, 320])
        distance2 = jnp.linalg.norm(unnorm2_pred - unnorm_label, axis=-1) # (B, T)
        p5_true = jnp.mean(distance2 <= 5.0) * 100 # no label mask for now
        p10_true = jnp.mean(distance2 <= 10.0) * 100
        p15_true = jnp.mean(distance2 <= 15.0) * 100
        p20_true = jnp.mean(distance2 <= 20.0) * 100
        p50_true = jnp.mean(distance2 <= 50.0) * 100
        metrics.update({f'{k}/p5_true': p5_true, f'{k}/p10_true': p10_true, f'{k}/p15_true': p15_true,
                        f'{k}/p20_true': p20_true, f'{k}/p50_true': p50_true})


    # Final loss
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    for k, v in losses.items():
      assert v.shape == (B, T), (k, v.shape, (B, T))
    final_loss = sum([v for k, v in losses.items()]) # (B, T)
    sum_label_mask = nn.f32(label_mask).sum()
    final_loss = jnp.where(sum_label_mask > 0, final_loss.sum() / sum_label_mask, 0.0) # average over valid labels
    metrics['gflops'] = self.gflops(carry, data)

    # metrics.update(self._metrics(data, logits))
    outs = {'preds': {k: jax.nn.tanh(v.pred()) for k, v in dists.items()}}
    carry = (enc_carry,)
    return final_loss, (carry, outs, metrics)

  def train(self, carry, data):
    opt_mets, (carry, outs, mets) = self.opt(self._loss, carry, data, training=True, has_aux=True)
    mets.update(opt_mets)
    return carry, outs, mets

  def report(self, carry, data):
    _, (carry, outs, metrics) = self._loss(carry, data, training=False)
    preds = outs['preds']
    valid_labels = data['label_mask'] # (B, T)

    B, T = data['is_first'].shape
    RB = min(6, B)

    # Video preds
    if self.centkey is not None:
      label = data[self.centkey][:RB]
      pred = preds[self.centkey][:RB]
      pred = unnormpupilcentroid(pred, self.label_space[self.centkey].high + 1).astype(jnp.uint32)
      valid_label = valid_labels[:RB]

      for key in self.imgkeys:
        true = nn.f32(data[key][:RB, ..., :1])
        true = jnp.repeat(true, 3, axis=-1) # grayscale to rgb
        # min max normalization, and take only the first channel
        true_min = true.min(axis=[2, 3], keepdims=True) # (RB, T, 1, 1, C)
        true_max = true.max(axis=[2, 3], keepdims=True) # (RB, T, 1, 1, C)
        norm = (true - true_min) / (true_max - true_min).clip(1e-8)
        norm = (norm * 255).clip(0, 255).astype(jnp.uint8)

        # draw label cross
        norm_with_label_1 = draw_cross_hair_with_dot(norm, label[..., 0], label[..., 1], color=ORANGE)
        norm_with_label_2 = norm
        norm_with_label = F.where(valid_label, norm_with_label_1, norm_with_label_2)

        # draw prediction cross
        norm = draw_cross_hair_with_dot(norm_with_label, pred[..., 0], pred[..., 1], color=BLUE)

        # video = jnp.concatenate([true, pred, error], 2)
        video = norm

        video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
        mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
        border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
        # TODO: for every prediction that match, make the border change its corresponding color
        # border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
        video = jnp.where(mask, video, border[None, :, None, None, :])
        video = jnp.concatenate([video, 0 * video[:, :10]], 1)

        _B, _T, _H, _W, _C = video.shape
        grid = video.transpose((1, 2, 0, 3, 4)).reshape((_T, _H, _B * _W, _C))
        metrics[f'openloop/video/{key}'] = grid

    return carry, {}, metrics




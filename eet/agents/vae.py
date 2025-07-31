"""
File: vae.py
Author: Viet Nguyen
Date: 2025-07-29

Description: develop VAE agent. Implementing
"""

from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import re

from lib.common import Space, print
from lib.agent.inactive import JAXAgent
from lib import nn
from lib.nn import functional as F

from modeling.encoder import Encoder
from modeling.decoder import Decoder
from .utils import *

class VAEAgent(JAXAgent):

  def __init__(self, obs_space, label_space, config, *args, **kwargs):
    self.obs_space = obs_space
    self.label_space = label_space
    exclude = ('is_first', 'is_last', 'is_dataset_first', 'is_dataset_last')
    enc_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.encoder.prockeys, k))}
    dec_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.decoder.prockeys, k))}
    self.encoder = Encoder(enc_space, **config.encoder.cnn, name="enc")
    self.dynamics = nn.MLPHead(**config.dynamics.mlp, name="dyn")
    self.decoder = Decoder(dec_space, **config.decoder.cnn, name="dec")
    _label_space = {k: Space(np.float32, v.shape, v.low, v.high) if ispupilcentroid(v) else v for k, v in label_space.items()}
    # print(f"[CNNAgent.__init__] _label_space: {_label_space}", color='green')
    self.head = nn.MLPHead(_label_space, {k: 'symlog_mse' for k in label_space}, **config.head.general, name='head')
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
    return ()

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()

  def infer(self, carry, obs, mode="train"):
    feat = self.encoder(obs, obs['is_first'], training=False, single=True) # (B, dim)
    dist = self.head(feat, bdims=1)
    result = {k: v.pred() for k, v in dist.items()}
    return carry, result, {}

  def _loss(self, data, training=True):
    metrics = {}
    reset = data['is_first']
    label_mask = data['label_mask'] # if the label is valid (B, T)

    feat = self.encoder(data, reset, training=training, single=False) # (B, dim)
    dists = self.head(feat, bdims=2)
    losses = {}
    for key, dist in dists.items():
      space, value = self.label_space[key], data[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = normpupilcentroid(nn.f32(value), space.high + 1) if ispupilcentroid(space) else value
      losses[key] = F.mask(dist.loss(nn.sg(target)), label_mask) # (B, T) x (B, T)
    # Assert shape
    B, T = reset.shape
    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    # Compute accuracy metrics
    for k, dist in dists.items():
      if ispupilcentroid(self.label_space[k]):
        pred = dist.pred()
        pred = unnormpupilcentroid(pred, self.label_space[k].high + 1)
        label = nn.f32(data[k])
        distance = jnp.linalg.norm(pred - label, axis=-1) # (B, T)
        # for some frame, the label might not be available, so we do this
        p5 = jnp.sum(F.mask(distance <= 5.0, label_mask)) / jnp.sum(label_mask) * 100
        p10 = jnp.sum(F.mask(distance <= 10.0, label_mask)) / jnp.sum(label_mask) * 100
        p15 = jnp.sum(F.mask(distance <= 15.0, label_mask)) / jnp.sum(label_mask) * 100
        p20 = jnp.sum(F.mask(distance <= 20.0, label_mask)) / jnp.sum(label_mask) * 100
        p50 = jnp.sum(F.mask(distance <= 50.0, label_mask)) / jnp.sum(label_mask) * 100
        metrics.update({f'{k}/p5': p5, f'{k}/p10': p10, f'{k}/p15': p15, f'{k}/p20': p20, f'{k}/p50': p50})

    # Final loss
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    # assert set(losses.keys()) == set(self.scales.keys()), (
    #     sorted(losses.keys()), sorted(self.scales.keys()))
    # final_loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
    final_loss = sum([(v.sum(-1) / nn.f32(label_mask).sum(-1)).mean() for k, v in losses.items()])

    # metrics.update(self._metrics(data, logits))
    outs = {'preds': {k: v.pred() for k, v in dists.items()}}
    return final_loss, (outs, metrics)

  def train(self, carry, data):
    opt_mets, (outs, mets) = self.opt(self._loss, data, training=True, has_aux=True)
    mets.update(opt_mets)
    return carry, outs, mets

  def report(self, carry, data):
    _, (outs, metrics) = self._loss(data, training=False)
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
        norm = (true - true.min()) / (true.max() - true.min())
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

    return carry, metrics




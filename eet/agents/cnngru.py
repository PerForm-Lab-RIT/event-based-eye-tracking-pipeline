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
import time

from lib.common import Space, print
from lib.agent.inactive import JAXAgent
from lib import nn
from lib.nn import functional as F
from lib import tree

from modeling.encoder import Encoder
from .utils import *

class CNNGRUAgent(JAXAgent):

  def __init__(self, obs_space, label_space, config, *args, **kwargs):
    self.obs_space = obs_space
    self.label_space = label_space
    self.config = config
    exclude = ('is_first', 'is_last', 'is_dataset_first', 'is_dataset_last')
    enc_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.encoder.prockeys, k))}
    self.encoder = Encoder(enc_space, **config.encoder.cnn, name="enc")
    self.dynamics = nn.GRU(**config.dynamics.gru, name='dyn')
    _label_space = {k: Space(np.float32, v.shape, v.low, v.high) if ispupilcentroid(v) else v for k, v in label_space.items()}
    # print(f"[CNNAgent.__init__] _label_space: {_label_space}", color='green')
    self.head = nn.MLPHead(_label_space, {k: 'identity' for k in label_space}, **config.head.general, name='head')
    modules = [self.encoder, self.dynamics, self.head]
    self.opt = nn.Optimizer(modules, **config.opt, name="opt")

    self.imgkeys = [k for k in obs_space if isimage(obs_space[k])]
    centkey = [k for k in label_space if ispupilcentroid(label_space[k])]
    assert len(centkey) <= 1
    if len(centkey) == 1:
      self.centkey = centkey[0]
    else:
      self.centkey = None

    self.grunits = config.dynamics.gru.units

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = Space(np.int32)
    spaces['stepid'] = Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(tree.flatdict(dict(
        dyn=Space(np.float32, self.grunits),
      )))
    return spaces

  def init_infer(self, batch_size):
    return (self.dynamics.initial(batch_size),)

  def init_train(self, batch_size):
    a = (self.dynamics.initial(batch_size),) # (B, Z)
    # print(f"[CNNGRU.init_train] init carry: {a[0].shape}", color='green')
    return a

  def init_report(self, batch_size):
    return (self.dynamics.initial(batch_size),)

  def infer(self, carry, obs, mode="train"):
    obs = preprocess_data(obs)
    (dyn_carry,) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    # print(f"[CNNGRU.infer] reset.shape: {reset}")
    # print(f"[CNNGRU.infer] obs: {jax.tree.map(lambda x: x.shape, obs)}")

    feat = self.encoder(obs, obs['is_first'], **kw) # (B, dim)
    # print(f"[CNNGRU.infer] feat.shape: {feat}")
    dyn_carry, feat = self.dynamics(nn.cast(dyn_carry), feat, reset, single=True) # (B, dim)
    dist = self.head(feat, bdims=1) # (B, dim)
    result = {k: jax.nn.tanh(v.pred()) for k, v in dist.items()}

    carry = (dyn_carry,)
    out = {}
    if self.config.replay_context:
      out.update(tree.flatdict(dict(dyn=dyn_carry)))
    return carry, result, out

  def _loss(self, carry, data, training=True):
    metrics = {}
    data = preprocess_data(data)
    reset = data['is_first']
    label_mask = data['label_mask'] # if the label is valid (B, T)
    (dyn_carry,) = carry # (B, dim)
    # print(f"[CNNGRUAgent._loss] dyn_carry.shape: {dyn_carry.shape}", color='green')
    B, T = reset.shape

    feat = self.encoder(data, reset, training=training, single=False) # (B, T, dim)
    dyn_carry, seq_carry = self.dynamics(nn.cast(dyn_carry), feat, reset, single=False) # (B, dim), (B, T, dim)
    # print((f"CNNGRUAgent._loss] seq_carry.shape: {seq_carry.shape}"), color='green')
    # Mobina: timing start
    start_time = time.time()

    #feat = self.encoder(data, reset, training=training, single=False) # (B, dim)
    #dists = self.head(feat, bdims=2)

    # Mobina-latency: jax.block_until_ready() forces JAX to finish
    # all async computation before we stop the timer. 
    #jax.block_until_ready(feat)

    # Mobina: timing end
    end_time = time.time()
    total_ms = (end_time - start_time) * 1000  # milliseconds
    metrics['latency_ms_per_frame'] = float(total_ms / (B * T))

    # Mobina [gflops/gmacs]: 
    # gflops_val = float(self.gflops(None, data))
    # metrics['gflops'] = gflops_val
    # metrics['gmacs'] = float(gflops_val / 2.0)

    # Mobina [params_m]: ninjax modules:
    # Parameters live in the global ninjax Context dict, keyed by path strings
    # like "enc/cnn0/kernel". The correct ninjax API to read them is .values,
    # which is a property defined on ninjax.Module that filters the global context
    # by this module's path prefix and returns a flat dict of {name: array}.
    enc_params  = jax.tree_util.tree_leaves(self.encoder.values)   
    head_params = jax.tree_util.tree_leaves(self.head.values)      
    all_params  = enc_params + head_params
    total_params = int(sum(np.prod(p.shape) for p in all_params if hasattr(p, 'shape')))
    metrics['params_m'] = float(total_params / 1e6)  # total params in millions


    dists = self.head(seq_carry, bdims=2) # (B, T, dim)
    losses = {}
    for key, dist in dists.items():
      space, value = self.label_space[key], data[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = normpupilcentroid(nn.f32(value), space.high + 1) if (ispupilcentroid(space) or key == 'pupil') else value
      prediction = jax.nn.tanh(dist.pred())
      _loss = ((prediction - nn.sg(target))**2).sum(-1) # (B, T) # MSE Loss
      losses[key] = F.mask(_loss, label_mask) # (B, T) x (B, T)
    # Assert shape
    #B, T = reset.shape
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

    # compute some metrics: gflops here: has to be a scalar value
    # metrics['gflops_of_things'] = new_number

    # Final loss
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    for k, v in losses.items():
      assert v.shape == (B, T), (k, v.shape, (B, T))
    final_loss = sum([v for k, v in losses.items()]) # (B, T)
    sum_label_mask = nn.f32(label_mask).sum()
    final_loss = jnp.where(sum_label_mask > 0, final_loss.sum() / sum_label_mask, 0.0) # average over valid labels

    # metrics.update(self._metrics(data, logits))
    outs = {'preds': {k: jax.nn.tanh(v.pred()) for k, v in dists.items()}}
    carry = (dyn_carry,)
    entries = (seq_carry,)
    return final_loss, (outs, carry, entries, metrics)

  def train(self, carry, data):
    # print(f"[CNNGRUAgent.train] carry: {carry}")
    data = self.populate_data(data)
    carry, data, stepid = self._apply_replay_context(carry, data)
    opt_mets, (outs, carry, entries, mets) = self.opt(self._loss, carry, data,
      training=True, has_aux=True)
    mets.update(opt_mets)
    if self.config.replay_context:
      updates = tree.flatdict(dict(
        stepid=stepid, dyn=entries[0]))
      B, T = data['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    return carry, outs, mets

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    data = self.populate_data(data)
    carry, data, stepid = self._apply_replay_context(carry, data)
    _, (outs, carry, entries, metrics) = self._loss(carry, data, training=False)
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

    outs = {}
    if self.config.replay_context:
      updates = tree.flatdict(dict(
        stepid=stepid, dyn=entries[0]))
      B, T = data['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    return carry, outs, metrics

  def _apply_replay_context(self, carry, data):
    stepid = data['stepid']
    _obs_space = (*self.obs_space.keys(), *self.label_space.keys())
    obs = {k: data[k] for k in _obs_space}
    ext_obs = {k: data[k] for k in self.ext_space.keys()}

    if not self.config.replay_context:
      return carry, {**obs, **ext_obs}, stepid

    K = self.config.replay_context
    nested = tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('dyn',)]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    # NOTE: take the last state of the context stream (not the whole stream)
    # rep_carry = (self.dynamics.truncate(lhs(entries[0]), dyn_carry),)
    rep_carry = (jax.tree.map(lambda x: x[:, -1], lhs(entries[0])),)
    # print(f"[CNNGRUAgent._apply_replay_context] rep_carry: {rep_carry}", color='green')
    rep_obs = {k: rhs(data[k]) for k in _obs_space}
    rep_ext_obs = {k: rhs(ext_obs[k]) for k in self.ext_space}
    rep_stepid = rhs(stepid)
    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, ext_obs, stepid = jax.tree.map(
      lambda normal, replay: nn.functional.where(first_chunk, replay, normal),
      (carry, rhs(obs), rhs(ext_obs), rhs(stepid)),
      (rep_carry, rep_obs, rep_ext_obs, rep_stepid))
    # print(f"[CNNGRUAgent._apply_replay_context] carry after apply: {carry[0].shape}", color='blue') # (B, dim)
    return carry, {**obs, **ext_obs}, stepid

  def populate_data(self, data: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
    # Just for populating data when initialize params
    B, T = data["is_first"].shape
    # for k, v in self.ext_space.items():
    #   if k not in data:
    #     _data = jnp.zeros(v.shape, dtype=v.dtype)
    #     _data = _data[None, None].repeat(B, axis=0).repeat(T, axis=1)
    #     data[k] = _data
    return data

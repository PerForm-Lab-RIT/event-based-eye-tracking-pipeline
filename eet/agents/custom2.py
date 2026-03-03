"""
File: custom2.py
Author: Viet Nguyen
Date: 2025-08-04

Description: develop my own custom model. Basically a model with integrated RSSM, similar to custom.py
But now, we modify so that it is more robust
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
from lib import tree

from modeling.encoder import Encoder
from modeling.ssm import RSSM, AIS2M
from .utils import *

class Custom2Agent(JAXAgent):

  def __init__(self, obs_space, label_space, config, *args, **kwargs):
    self.config = config
    self.obs_space = obs_space
    self.label_space = label_space
    exclude = ('is_first', 'is_last', 'is_dataset_first', 'is_dataset_last')
    enc_space = {k: v for k, v in obs_space.items() if (k not in exclude and re.match(config.encoder.prockeys, k))}
    _label_space = {k: Space(np.float32, v.shape, v.low, v.high) if ispupilcentroid(v) else v for k, v in label_space.items()}
    # print(f"[CustomAgent.__init__] _label_space: {_label_space}", color='green')
    self.encoder = Encoder(enc_space, **config.encoder.cnn, name="enc")
    self.dynamics = AIS2M(**config.dynamics.ais2m, name="dyn")
    self.head = nn.MLPHead(_label_space, {k: 'identity' for k in label_space}, **config.head.general, name='head')
    modules = [self.encoder, self.dynamics, self.head]
    self.opt = nn.Optimizer(modules, **config.opt, name="opt")

    self._feat2vec = lambda x: jnp.concatenate([
      nn.cast(x['deter']),
      nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1))),
      nn.cast(x['combine'].reshape((*x['combine'].shape[:-2], -1))),
    ], axis=-1)

    self.imgkeys = [k for k in obs_space if isimage(obs_space[k])]
    centkey = [k for k in label_space if ispupilcentroid(label_space[k])]
    assert len(centkey) <= 1
    if len(centkey) == 1:
      self.centkey = centkey[0]
    else:
      self.centkey = None

    self.loss_scales = config.loss_scales if hasattr(config, 'loss_scales') else {}

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = Space(np.int32)
    spaces['stepid'] = Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(tree.flatdict(dict(
        dyn=self.dynamics.entry_space,
      )))
    return spaces

  def init_infer(self, batch_size):
    return (self.dynamics.initial(batch_size),)

  def init_train(self, batch_size):
    return (self.dynamics.initial(batch_size),)

  def init_report(self, batch_size):
    return (self.dynamics.initial(batch_size),)

  def infer(self, carry, obs, mode="train"):
    (dyn_carry,) = carry
    obs = preprocess_data(obs)
    reset = obs['is_first']
    # infer
    embed = self.encoder(obs, reset, training=False, single=True) # (B, dim)
    carry, dyn_entry, feat, postlogit, priorlogit, alphalogit = self.dynamics.observe(nn.cast(dyn_carry), embed, reset, training=mode=='train', single=True) # (B, dim)
    dist = self.head(self._feat2vec(feat), bdims=1)
    result = {k: jnp.clip(v.pred(), -1, 1) for k, v in dist.items()}
    carry = (dyn_carry,)
    outs = {}
    if self.config.replay_context:
      outs.update(tree.flatdict(dict(dyn=dyn_entry)))
    return carry, result, outs

  def _loss(self, carry, data, training=True):
    metrics = {}
    (dyn_carry,) = carry
    target_alpha: jax.Array = nn.sg(self._compute_confidence(data)) # confidence is computed before data normalization
    data = preprocess_data(data)
    reset = data['is_first']
    label_mask = data['label_mask'] # if the label is valid (B, T)
    # actual computation
    embed = self.encoder(data, reset, training=training, single=False) # (B, dim)
    # Forward and KL losses
    metrics['confidence'] = target_alpha.reshape(-1)
    dyn_carry, dyn_entries, los, feat, mets = self.dynamics.loss(dyn_carry, embed, target_alpha, reset, training)
    metrics.update(mets)
    dists = self.head(self._feat2vec(feat), bdims=2)
    losses = {}
    losses.update(los)
    # Accuracy loss
    for key, dist in dists.items():
      space, value = self.label_space[key], data[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = normpupilcentroid(nn.f32(value), space.high + 1) if (ispupilcentroid(space) or key == 'pupil') else value
      prediction = jnp.clip(dist.pred(), -1, 1)
      _loss = F.huber_loss(prediction, nn.sg(target)).sum(-1) # (B, T)
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
    final_loss = sum([v * self.loss_scales.get(k, 1.0) for k, v in losses.items()]) # (B, T)
    sum_label_mask = nn.f32(label_mask).sum()
    final_loss = jnp.where(sum_label_mask > 0, final_loss.sum() / sum_label_mask, 0.0) # average over valid labels

    # metrics.update(self._metrics(data, logits))
    outs = {'preds': {k: jnp.clip(v.pred(), -1, 1) for k, v in dists.items()}}
    carry = (dyn_carry,)
    entries = (dyn_entries,)
    return final_loss, (outs, carry, entries, metrics)

  def _compute_confidence(self, data):
    """
    Compute signal quality metric for RAW (un-normalized) event frames.

    For event-based eye tracking, this function computes a signal quality score that
    serves as the alpha parameter for the AIS2M dynamics model. This function operates
    on RAW event data BEFORE z-score normalization, allowing us to use actual event
    magnitudes and traditional signal-to-noise concepts.

    Event-Based Signal Quality Methodology:
    =======================================
    Event cameras generate sparse temporal change data. Signal quality is determined by:
    1. Event density ratio: More events in pupil region vs background

    Args:
        data (Dict[str, jax.Array]): Batch data containing:
            - 'frame': (B, T, H, W, 2) RAW (un-normalized) event frames
            - 'pupil': (B, T, 2) pupil center coordinates in [y, x] format
            - 'label_mask': (B, T) boolean mask for valid pupil labels

    Returns:
        jax.Array: Signal quality scores with shape (B, T)
            - Values in range [0, 1] where:
              * 0 = poor signal quality (low event density ratio)
              * 1 = excellent signal quality (high event density ratio)
              * default_alpha used for frames with invalid labels

    Configuration Parameters:
        - config.snr_box_size: Tuple (width, height) for signal region box size
        - config.default_alpha: Alpha value for frames with invalid labels
        - config.event_threshold: Threshold for significant events (default: 0.01)

    Signal Quality Metric:
    =====================
    Event Density Ratio:
       - Ratio of event density in pupil region vs background
       - Higher ratio indicates more eye movement activity in signal region

    Shape Flow:
    ==========
    frame: (B, T, H, W, 2) → signal_mask: (B, T, H, W, 2) → metrics: (B, T) → snr: (B, T)
    """
    # Compute signal quality for RAW (un-normalized) EVENT frames
    # Raw frames contain actual event magnitudes and sparse data
    frame = data['frame']  # (B, T, H, W, 2) - RAW event frame
    # Assume that the frame is event frame (0.5 for no events, 0 and 1 for negative and positive events)
    frame = jnp.abs(frame - 0.5)  # Normalize to [0, 1] range
    label = data['pupil']  # (B, T, 2) in y, x format
    label_mask = data['label_mask']  # (B, T) - valid label mask

    B, T, H, W, C = frame.shape

    # Define signal region box size around pupil center
    box_size_width, box_size_height = self.config.confidence.get('roi_size', (80, 40))  # default box size
    assert box_size_width != 0 and box_size_height != 0, "Box size must be non-zero"

    # Event threshold for significant events (raw data typically has different scale)
    event_threshold = self.config.confidence.get('event_threshold', 0.0)

    # Ensure label coordinates are within frame bounds
    y_coords = jnp.clip(label[..., 0], 0, H - 1).astype(jnp.int32)  # y coordinates
    x_coords = jnp.clip(label[..., 1], 0, W - 1).astype(jnp.int32)  # x coordinates

    # Define signal region bounds
    half_box_width = box_size_width // 2
    half_box_height = box_size_height // 2
    y_min = jnp.clip(y_coords - half_box_height, 0, H - 1)
    y_max = jnp.clip(y_coords + half_box_height, 0, H - 1)
    x_min = jnp.clip(x_coords - half_box_width, 0, W - 1)
    x_max = jnp.clip(x_coords + half_box_width, 0, W - 1)

    # Create coordinate grids for spatial masking
    y_grid, x_grid = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    y_grid = y_grid[None, None, :, :, None]  # (1, 1, H, W, 1)
    x_grid = x_grid[None, None, :, :, None]  # (1, 1, H, W, 1)

    # Create signal region mask for each frame in the batch
    y_min_expanded = y_min[..., None, None, None]  # (B, T, 1, 1, 1)
    y_max_expanded = y_max[..., None, None, None]  # (B, T, 1, 1, 1)
    x_min_expanded = x_min[..., None, None, None]  # (B, T, 1, 1, 1)
    x_max_expanded = x_max[..., None, None, None]  # (B, T, 1, 1, 1)

    signal_mask = (
        (y_grid >= y_min_expanded) & (y_grid <= y_max_expanded) &
        (x_grid >= x_min_expanded) & (x_grid <= x_max_expanded)
    )  # (B, T, H, W, 1)

    # Expand signal mask to match frame channels
    signal_mask = jnp.broadcast_to(signal_mask, (B, T, H, W, C))  # (B, T, H, W, C)
    noise_mask = ~signal_mask

    # ============================================================================
    # Computing confidence based on signal quality and coverage
    # ============================================================================

    # Method 1: Count pixels with significant events using configurable threshold
    signal_event_count = jnp.where(signal_mask, jnp.abs(frame) > event_threshold, 0).sum(axis=(2, 3, 4))  # (B, T)
    noise_event_count = jnp.where(noise_mask, jnp.abs(frame) > event_threshold, 0).sum(axis=(2, 3, 4))  # (B, T)
    # Use absolute signal event count (simpler and more direct)
    signal_strength = signal_event_count / (noise_event_count + 1e-8)  # normalize by max possible pixels # (B, T)
    # signal_quality = np.clip(signal_strength, 0, 1)  # ensure [0, 1] range
    # This is often referred to as "signal-to-noise ratio" (SNR) in signal processing
    signal_quality = jax.nn.sigmoid(signal_strength) # (B, T)

    # Methoid 2: Calculate total number of signal events within the interested region (signal_mask area)
    total_signal_region_pixels = signal_mask.sum(axis=(2, 3, 4))  # Total pixels in signal region # (B, T)
    # Calculate signal density within the interested region
    signal_coverage_ratio = signal_event_count / total_signal_region_pixels # (B, T)
    signal_coverage_ratio = signal_coverage_ratio / self.config.confidence.get('event_coverage_ratio_threshold', 0.1) # (B, T)
    signal_coverage_ratio = jnp.clip(signal_coverage_ratio, 0, 1) # ensure [0, 1] range # (B, T)

    # Finally, compute the final confidence score
    confidence = self.config.confidence.weights.get('quality', 0.5) * signal_quality\
      + self.config.confidence.weights.get('coverage', 0.5) * signal_coverage_ratio

    # Apply label mask - use default alpha for frames with invalid pupil labels
    default_alpha = self.config.confidence.get('default_alpha', 0.5)
    confidence = jnp.where(label_mask, confidence, default_alpha)  # (B, T)

    return confidence  # Shape: (B, T) - one confidence score per frame

  def train(self, carry, data):
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

    (dyn_carry,) = carry # prevact is one previous actions
    carry = (dyn_carry,)
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
    rep_carry = (
      self.dynamics.truncate(lhs(entries[0]), dyn_carry),
    )
    rep_obs = {k: rhs(data[k]) for k in _obs_space}
    rep_ext_obs = {k: rhs(ext_obs[k]) for k in self.ext_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, ext_obs, stepid = jax.tree.map(
      lambda normal, replay: nn.functional.where(first_chunk, replay, normal),
      (carry, rhs(obs), rhs(ext_obs), rhs(stepid)),
      (rep_carry, rep_obs, rep_ext_obs, rep_stepid))
    return carry, {**obs, **ext_obs}, stepid
from ..processor import BaseProcessor
from ..types import BatchItemDataType, EyeEpisodeDataType
from .event_transform import build_event_transform, EventTransform
from ...utils import Space

import numpy as np
import torch

class GeneralProcessor(BaseProcessor):

  def __init__(
    self,
    event_repr: str,
    width: int,
    height: int,
    mode: str = "train",
    chw: bool = False,
    aug_flip_x_prob: float = 0.0,
    aug_flip_y_prob: float = 0.0,
    aug_rotate_prob: float = 0.0,
    aug_rotate_max_deg: int = 0,
  ):
    self.event_transform: EventTransform = build_event_transform(event_repr, width=width, height=height)
    self.width = width
    self.height = height
    self.mode = str(mode).lower()
    self.chw = chw
    self.aug_flip_x_prob = aug_flip_x_prob
    self.aug_flip_y_prob = aug_flip_y_prob
    self.aug_rotate_prob = aug_rotate_prob
    self.aug_rotate_max_deg = aug_rotate_max_deg

  def __call__(self, item: BatchItemDataType) -> None:
    results = {
      "input": [],
      "label": []
    }

    default_height = item.height
    default_width = item.width

    # Pre-compute rescale factors once
    needs_rescale = (self.height != default_height or self.width != default_width)
    if needs_rescale:
      scale_y = self.height / default_height
      scale_x = self.width / default_width

    # Decide augmentations once per item (applied consistently to all timebins)
    is_event = item.modality == 'event'
    is_train = self.mode == 'train'
    do_flip_x = is_event and is_train and np.random.random() < self.aug_flip_x_prob
    do_flip_y = is_event and is_train and np.random.random() < self.aug_flip_y_prob
    do_rotate = is_event and is_train and np.random.random() < self.aug_rotate_prob
    rotate_angle = int(np.random.randint(-self.aug_rotate_max_deg, self.aug_rotate_max_deg)) if do_rotate else 0

    # Process all timebins/frames
    for t in range(item.batch_length):
      if item.modality == 'event':
        label = item.labels[t][:2].copy().astype(np.float32)  # [y, x]
        data_timebin = item.data[t]

        if do_flip_x:
          data_timebin, label = self._aug_flip_x(data_timebin, label, default_width, default_height)
        if do_flip_y:
          data_timebin, label = self._aug_flip_y(data_timebin, label, default_width, default_height)
        if do_rotate:
          data_timebin, label = self._aug_rotate(data_timebin, label, default_width, default_height, rotate_angle)

        if needs_rescale:
          label = label * np.array([scale_y, scale_x], dtype=np.float32)
          label = label.clip(0, np.array([self.height - 1, self.width - 1], dtype=np.float32))
        results['label'].append(torch.as_tensor(label.astype(np.int32), dtype=torch.long))

        frame = self.event_transform(
          data_timebin,
          original_width=default_width,
          original_height=default_height
        )
        results['input'].append(torch.as_tensor(frame, dtype=torch.float32))
      else:
        frame = item.data[t]
        label = item.labels[t]
        results['input'].append(torch.as_tensor(frame, dtype=torch.float32))
        results['label'].append(torch.as_tensor(label))

    results['input'] = torch.stack(results['input'], dim=0)
    results['label'] = torch.stack(results['label'], dim=0)

    if not item.temporal and results['input'].shape[0] == 1:
      results['input'] = results['input'][0]
      results['label'] = results['label'][0]

    if self.chw:
      if item.temporal:
        if results['input'].ndim == 4:
          results['input'] = results['input'].permute(0, 3, 1, 2)
      else:
        if results['input'].ndim == 3:
          results['input'] = results['input'].permute(2, 0, 1)

    if item.source_path is not None:
      results['source_path'] = item.source_path

    # Store processed results in the item
    item.processed = results

  @property
  def input_space(self) -> Space:
    """Return the input space of the dataset."""
    channel = self.event_transform.output_channels
    return Space(
      shape=(self.height, self.width, channel) if not self.chw else (channel, self.height, self.width),
      dtype=np.float32
    )

  @property
  def label_space(self) -> Space:
    """Return the label space of the dataset."""
    return Space(shape=(2,), dtype=np.int32)  # label: [y, x]

  def _aug_flip_x(self, events: np.ndarray, label: np.ndarray, width: int, height: int):
    """Flip events and label horizontally."""
    events = events.copy()
    events['x'] = (width - 1) - events['x']
    label = label.copy()
    label[1] = (width - 1) - label[1]  # label is [y, x]
    return events, label

  def _aug_flip_y(self, events: np.ndarray, label: np.ndarray, width: int, height: int):
    """Flip events and label vertically."""
    events = events.copy()
    events['y'] = (height - 1) - events['y']
    label = label.copy()
    label[0] = (height - 1) - label[0]  # label is [y, x]
    return events, label

  def _aug_rotate(self, events: np.ndarray, label: np.ndarray, width: int, height: int, angle_deg: int):
    """Rotate events and label around the image center, clipping out-of-bounds events."""
    angle = angle_deg / 180.0 * np.pi
    cx, cy = width / 2.0, height / 2.0

    x = events['x'].astype(np.float32)
    y = events['y'].astype(np.float32)
    rx = (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle) + cx
    ry = (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle) + cy

    mask = (rx >= 0) & (rx < width) & (ry >= 0) & (ry < height)
    events = events[mask].copy()
    events['x'] = rx[mask].astype(events.dtype['x'].type)
    events['y'] = ry[mask].astype(events.dtype['y'].type)

    label = label.copy()
    lx, ly = float(label[1]), float(label[0])
    label[1] = (lx - cx) * np.cos(angle) - (ly - cy) * np.sin(angle) + cx
    label[0] = (lx - cx) * np.sin(angle) + (ly - cy) * np.cos(angle) + cy
    label = label.clip(0, np.array([height - 1, width - 1], dtype=np.float32))
    return events, label

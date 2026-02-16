
from ..processor import BaseProcessor
from ..types import BatchItemDataType
from .event_transform import build_event_transform

import numpy as np
import torch

class GeneralProcessor(BaseProcessor):

  def __init__(self, event_repr: str, width: int, height: int, chw: bool = False):
    self.event_transform = build_event_transform(event_repr, width=width, height=height)
    self.width = width
    self.height = height
    self.chw = chw

  def __call__(self, item: BatchItemDataType) -> None:
    results = {
      "event": [],
      "label": []
    }

    default_height = item.height
    default_width = item.width

    # Pre-compute rescale factors once
    needs_rescale = (self.height != default_height or self.width != default_width)
    if needs_rescale:
      scale_y = self.height / default_height
      scale_x = self.width / default_width

    # Process all timebins
    for t in range(item.batch_length):
      # rescale label
      label = item.labels[t][:2].copy()  # only takes the first two values, copy to avoid modifying original
      # NOTE: rescale label if needed
      if needs_rescale:
        label = (label * np.array([scale_y, scale_x])).astype(np.int32)
        label = label.clip(0, np.array([self.height - 1, self.width - 1], dtype=np.int32))
      results['label'].append(torch.as_tensor(label, dtype=torch.long))

      # process event transform
      event_timebin = item.events[t]
      frame = self.event_transform(
        event_timebin,
        original_width=default_width,
        original_height=default_height
      )
      results['event'].append(torch.as_tensor(frame, dtype=torch.float32))

    results['event'] = torch.stack(results['event'], dim=0) # (T, H, W, C)
    results['label'] = torch.stack(results['label'], dim=0) # (T, 2)

    if self.chw:
      results['event'] = results['event'].permute(0, 3, 1, 2)  # (T, C, H, W)

    # Store processed results in the item
    item.processed = results






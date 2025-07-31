"""
File: eveye.py
Author: Viet Nguyen
Date: 2025-07-28

Description: Data loading environment for EV-Eye Dataset
NOTE: Label space for pupil centroid has to be uint32. E.g.,
  'pupil': Space(dtype=np.int32, shape=(2,), low=0, high=[DEFAULT_IMAGE_HEIGHT - 1, DEFAULT_IMAGE_WIDTH - 1]), # y, x
"""

import functools
import sys, pathlib
from typing import Dict, List, Tuple, Iterator
import numpy as np
import h5py
import pandas as pd
from PIL import Image

from lib import print, Space
from lib.envs.base import InactiveEnv

from .event_transform import EventTransform, EVENT_DTYPE, build_event_transform

DEFAULT_TIMEBIN = 40000 # default using micro second unit, 40 ms
DEFAULT_WIDTH = 346
DEFAULT_HEIGHT = 260

class EVEyeEnv(InactiveEnv):
  def __init__(self, datadir: pathlib.Path | str,
      event_transform: str = "voxelgrid",
      image_size: tuple = (DEFAULT_WIDTH, DEFAULT_HEIGHT), seed: int = 10,
      timebin: int = DEFAULT_TIMEBIN,
      mode: str = "train", train_ratio: float = 0.9) -> None:
    """Initialization

    Args:
        datadir (pathlib.Path | str): _description_
        event_transform (str, optional): _description_. Defaults to "voxelgrid".
        image_size (tuple, optional): _description_. Defaults to (346, 260).
        seed (int, optional): _description_. Defaults to 10.
        timebin (int, optional): _description_. Defaults to DEFAULT_TIMEBIN.
        mode (str): 'train' or 'eval' mode
        train_ratio (float): ratio of data to use for training (0.0 to 1.0)
    """

    self.datadir = pathlib.Path(datadir) / 'Data_davis'
    self.event_transform = event_transform
    self.image_size = image_size
    self.timebin = timebin
    self.mode = mode
    self.train_ratio = train_ratio

    if seed is None:
      self.seed = np.random.randint(0, 1000000)
    else:
      self.seed = seed
    self.rng = np.random.default_rng(self.seed)

    assert mode in ['train', 'eval'], f"mode must be 'train' or 'eval', got {mode}"
    assert 0.0 <= train_ratio <= 1.0, f"train_ratio must be between 0.0 and 1.0, got {train_ratio}"

    # Load the event transform
    self.event_transform_fn = build_event_transform(event_transform, width=image_size[0], height=image_size[1])

    # Collect all sequence paths
    all_paths = []
    for user_dir in self.datadir.glob("*"):
      # print(f"user dir: {user_dir}")
      if not user_dir.is_dir():
        continue
      for eye_side_dir in user_dir.glob("*"):
        if not eye_side_dir.is_dir():
          continue
        for session_dir in eye_side_dir.glob("*"):
          if not session_dir.is_dir():
            continue
          # print(f"session dir: {session_dir}")
          event_file = session_dir / "events" / "events.txt"
          # label_file = session_dir / f"{user_dir.name}_{eye_side_dir.name.split('_')[0] if '_' in eye_side_dir.name else '1'}.csv"
          # Find the CSV label file in the session directory
          label_files = list(session_dir.glob("*.csv"))
          if len(label_files) > 0:
            label_file = label_files[0]
          else:
            continue
          if event_file.exists() and label_file.exists():
            all_paths.append((event_file, label_file))

    # Split data into train and eval sets based on a deterministic seed
    # Use a fixed seed for consistent splits across different runs
    split_rng = np.random.default_rng(42)  # Fixed seed for consistent train/eval splits
    indices = split_rng.permutation(len(all_paths))
    n_train = int(len(all_paths) * self.train_ratio)

    if self.mode == 'train':
      self.data_indices = indices[:n_train]
    else:  # eval mode
      self.data_indices = indices[n_train:]

    # Filter all_paths to only include the appropriate subset
    self.all_paths = [all_paths[i] for i in self.data_indices]

    self.width = image_size[0]
    self.height = image_size[1]

    # Initialize state tracking
    self._reset_dataset()
    self._reset_seq()

  def _load_event_file(self, event_file: pathlib.Path) -> np.ndarray:
    """Load events from text file"""
    # Load raw event data (t, x, y, p)
    data = np.loadtxt(event_file)

    # Convert to structured array with proper dtype
    events = np.zeros(len(data), dtype=[
      ('t', np.int64),
      ('x', np.int32),
      ('y', np.int32),
      ('p', np.int8)
    ])
    events['t'] = data[:, 0].astype(np.int64)
    events['x'] = data[:, 1].astype(np.int32)
    events['y'] = data[:, 2].astype(np.int32)
    events['p'] = data[:, 3].astype(np.int8)

    return events

  def _load_label_file(self, label_file: pathlib.Path) -> np.ndarray:
    """Load labels from CSV file and extract ellipse parameters"""
    import json

    df = pd.read_csv(label_file)

    # Filter rows that have region_shape_attributes
    labeled_rows = df[df['region_shape_attributes'].notna()]

    labels = []
    for _, row in labeled_rows.iterrows():
      # Extract timestamp from filename (assuming format: framenum_timestamp.png)
      filename = row['filename']
      timestamp = int(filename.split('_')[1].split('.')[0])

      # Parse ellipse parameters from region_shape_attributes
      region_attrs = json.loads(row['region_shape_attributes'])
      if 'cx' in region_attrs and 'cy' in region_attrs:
        x_center = region_attrs['cx']
        y_center = region_attrs['cy']
        a_major = region_attrs.get('rx', 0)
        b_minor = region_attrs.get('ry', 0)
        rotation = region_attrs.get('theta', 0)

        labels.append([timestamp, x_center, y_center, a_major, b_minor, rotation])
    # Return array of shape (n_inferences, 6)
    return np.array(labels, dtype=np.float64)

  def _infer_label(self, timestamp: int, window: int=40000) -> Tuple[np.ndarray, bool]:
    """Infer label, the label is taken if it lies in a window of +- 40000 microseconds
      around the searchsorted timestamp
    """
    if len(self.labels) == 0:
      return np.zeros(5, dtype=np.float32), False  # [x, y, a, b, r]

    timestamps = self.labels[:, 0]
    # Find closest timestamp
    closest_idx = np.argmin(np.abs(timestamps - timestamp))
    closest_time = timestamps[closest_idx]

    # Check if within window
    if abs(closest_time - timestamp) <= window:
      return self.labels[closest_idx, 1:].astype(np.float32), True
    else:
      return np.zeros(5, dtype=np.float32), False

  def _reset_seq(self):
    """Reset to next sequence"""
    self.current_seq_id += 1
    self.current_step = 0
    self.is_first = True
    self.is_last = False

    # Load data for current sequence
    event_file, label_file = self.all_paths[self.current_seq_idx[self.current_seq_id]]
    self.event_seq = self._load_event_file(event_file)
    self.labels = self._load_label_file(label_file)

    self.current_event_id = 0
    # Start from first event timestamp
    self.current_time = self.event_seq['t'][0] + self.timebin if len(self.event_seq) > 0 else self.timebin

  def _reset_dataset(self):
    """Reset the whole dataset"""
    self.current_seq_id = -1
    self.current_seq_idx = self.rng.permutation(np.arange(len(self.all_paths)))
    self.is_dataset_first = True
    self.is_dataset_last = False

  @functools.cached_property
  def obs_space(self):
    frame_space = self.event_transform_fn.transform_space
    return {
      'frame': frame_space,
      'is_first': Space(dtype=bool),
      'is_last': Space(dtype=bool),
      'is_dataset_first': Space(dtype=bool),
      'is_dataset_last': Space(dtype=bool),
      'label_mask': Space(dtype=bool)
    }

  @functools.cached_property
  def label_space(self):
    return {
      # 'ellipse': Space(dtype=np.float32, shape=(2,)), # [x_center, y_center]  # [x_center, y_center, a_major, b_minor, rotation]
      'pupil': Space(dtype=np.int32, shape=(2,), low=0, high=[self.height - 1, self.width - 1]), # y, x
    }

  def get_dataset_info(self) -> Dict[str, int]:
    """Get information about the dataset split.

    Returns:
        Dict containing dataset information including total sequences,
        train sequences, eval sequences, and current mode sequences.
    """
    # We need to recalculate total sequences by counting all available sequences
    # This is a bit inefficient but ensures accuracy
    total_sequences = 0
    for user_dir in self.datadir.glob("*"):
      if not user_dir.is_dir():
        continue
      for eye_side_dir in user_dir.glob("*"):
        if not eye_side_dir.is_dir():
          continue
        for session_dir in eye_side_dir.glob("*"):
          if not session_dir.is_dir():
            continue
          event_file = session_dir / "events" / "events.txt"
          label_files = list(session_dir.glob("*.csv"))
          if len(label_files) > 0 and event_file.exists():
            total_sequences += 1

    n_train = int(total_sequences * self.train_ratio)
    n_eval = total_sequences - n_train
    current_sequences = len(self.all_paths)

    return {
      'total_sequences': total_sequences,
      'train_sequences': n_train,
      'eval_sequences': n_eval,
      'current_mode': self.mode,
      'current_sequences': current_sequences,
      'train_ratio': self.train_ratio,
    }

  @property
  def num_sequences(self) -> int:
    """Get the number of sequences in the current mode."""
    return len(self.all_paths)

  def step(self) -> Dict[str, np.ndarray]:
    """Step through one timebin of data"""
    if self.is_dataset_last:
      self._reset_dataset()
    if self.is_last:
      self._reset_seq()

    # Get events in current timebin
    next_event_id = np.searchsorted(self.event_seq['t'], self.current_time)
    events = self.event_seq[self.current_event_id:next_event_id]

    # Get interpolated label for current timestamp
    label, valid = self._infer_label(self.current_time)

    # Check boundary conditions
    is_first = self.current_step == 0
    is_last = next_event_id >= len(self.event_seq) - 1
    is_dataset_first = self.current_seq_id == 0
    is_dataset_last = self.current_seq_id >= len(self.all_paths) - 1  # Use filtered dataset length

    obs = self._obs(events, label, valid, is_first, is_last, is_dataset_first, is_dataset_last)

    # Update state
    self.is_first = is_first
    self.is_last = is_last
    self.is_dataset_first = is_dataset_first
    self.is_dataset_last = is_dataset_last
    self.current_event_id = next_event_id
    self.current_time += self.timebin
    self.current_step += 1

    return obs

  def _obs(self, events: np.ndarray, label: np.ndarray, valid: bool, is_first: bool,
      is_last: bool, is_dataset_first: bool, is_dataset_last: bool) -> Dict[str, np.ndarray]:
    """Generate observation dictionary"""
    # Transform events to frame representation
    frame = self.event_transform_fn(events, original_width=346, original_height=260)

    # Scale ellipse parameters to target image size if needed
    if self.width != 346 or self.height != 260:
      scaled_label = label.copy()
      scaled_label[0] = scaled_label[0] / 346 * self.width  # x_center
      scaled_label[1] = scaled_label[1] / 260 * self.height  # y_center
      scaled_label[2] = scaled_label[2] / 346 * self.width  # a_major
      scaled_label[3] = scaled_label[3] / 260 * self.height  # b_minor
      # rotation stays the same
    else:
      scaled_label = label
    prep_label = np.asarray([scaled_label[1], scaled_label[0]], dtype=np.int32)

    return {
      'frame': frame,
      'pupil': prep_label,
      'is_first': is_first,
      'is_last': is_last,
      'is_dataset_first': is_dataset_first,
      'is_dataset_last': is_dataset_last,
      'label_mask': valid,
    }

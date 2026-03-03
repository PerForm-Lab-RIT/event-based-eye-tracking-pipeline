"""
File: threeet.py
Author: Viet Nguyen
Date: 2025-05-30

Description: Data loading environment for ThreeET Dataset
NOTE: Label space for pupil centroid has to be uint32. E.g.,
  'pupil': Space(dtype=np.int32, shape=(2,), low=0, high=[DEFAULT_IMAGE_HEIGHT - 1, DEFAULT_IMAGE_WIDTH - 1]), # y, x

The environment supports train and eval modes. Data is split internally based on a deterministic seed (42)
to ensure consistent train/eval splits across different runs. The split is controlled by the train_ratio parameter.
"""

import functools
import sys, pathlib
from typing import Dict, List, Tuple, Iterator
import numpy as np
import h5py

from lib import print, Space
from lib.envs.base import InactiveEnv

from .event_transform import EventTransform, EVENT_DTYPE, build_event_transform


DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_TIMEBIN = 10000 # 10ms


def load_label_file(file_path: str) -> np.ndarray:
  """Load a label.txt file containing (x, y, label) coordinates.
  Args:
      file_path: Path to the label.txt file

  Returns:
      numpy array of shape (N, 3) containing (x, y, label) coordinates
  """
  # Read the file and convert each line to a tuple of integers
  data = []
  with open(file_path, 'r') as f:
    for line in f:
      # Remove parentheses and split by comma
      line = line.strip().strip('()').split(',')
      # Convert to integers
      x, y, label = map(int, line)
      data.append([y, x, label])
  # Convert to numpy array
  return np.array(data, dtype=np.int32)


# def event_to_binary_representation(events: np.ndarray, width: int, height: int, microbins: int = 4) -> np.ndarray:
#   """Convert events to binary representation.

#   Args:
#       events: numpy array of shape (N, 4) containing (t, x, y, p)

#   Returns:
#       numpy array of shape (H, W, 2) containing (positive, negative)
#   """
#   frame = np.zeros((height, width, 2), dtype=np.uint8)
#   for event in events:
#     print(event)
#     x, y, p = event['x'], event['y'], event['t']
#     frame[y, x, p] += 1
#   return frame


class ThreeETEnv(InactiveEnv):

  def __init__(self, datadir: pathlib.Path | str, event_transform: str,
      timebin: int = DEFAULT_TIMEBIN, image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
      seed=None, mode: str = 'train', train_ratio: float = 0.9):
    """_summary_

    Args:
      datadir (pathlib.Path): path to the data directory
      timebin (int, optional): in nanoseconds. Defaults to 10000 (10ms).
      mode (str): 'train' or 'eval' mode
      train_ratio (float): ratio of data to use for training (0.0 to 1.0)
    """
    if seed is None:
      # self.seed = np.random.randint(0, 1000000)
      self.seed = 42
    else:
      self.seed = seed
    self.mode = mode
    self.train_ratio = train_ratio
    self.timebin = timebin
    self.rng = np.random.default_rng(self.seed)
    eventdir = pathlib.Path(datadir) / "event_data" / "event_data" / "train"
    self.seq_dirs: List[pathlib.Path] = [] # e.g., `data/event-based-eye-tracking-ais2024/event_data/event_data/train/6_2`
    for seq_dir in eventdir.glob("*"):
      self.seq_dirs.append(seq_dir)

    assert DEFAULT_TIMEBIN % timebin == 0, f"timebin must be a divisor of DEFAULT_TIMEBIN={DEFAULT_TIMEBIN}"
    assert mode in ['train', 'eval'], f"mode must be 'train' or 'eval', got {mode}"
    assert 0.0 <= train_ratio <= 1.0, f"train_ratio must be between 0.0 and 1.0, got {train_ratio}"

    self.all_paths = []

    # Loop through all seuqences and files and do some mapping
    for seq_dir in self.seq_dirs:
      seq_name = str(seq_dir).split("/")[-1]
      data_path = seq_dir / f"{seq_name}.h5"
      label_path = seq_dir / f"label.txt"
      self.all_paths.append((data_path, label_path))

    # Split data into train and eval sets based on a deterministic seed
    # Use a fixed seed for consistent splits across different runs
    split_rng = np.random.default_rng(42)  # Fixed seed for consistent train/eval splits
    indices = split_rng.permutation(len(self.all_paths))
    n_train = int(len(self.all_paths) * self.train_ratio)

    if self.mode == 'train':
      self.data_indices = indices[:n_train]
    else:  # eval mode
      self.data_indices = indices[n_train:]

    # Filter all_paths to only include the appropriate subset
    self.all_paths = [self.all_paths[i] for i in self.data_indices]

    self.height = image_size[1]
    self.width = image_size[0]

    # NOTE: For event transform, we need to keep the original width and height to process event data onto it
    # After process event we downscale later
    self.event_transform = build_event_transform(event_transform, width=self.width, height=self.height)

    # Keep track of states
    # We call in this specific order to ensure that the state is reset correctly
    self._reset_dataset()
    self._reset_seq()

  def _reset_seq(self):
    """Reset a sequence

    Returns:
        _type_: _description_
    """
    self.current_seq_id += 1 # increment the sequence id, go to the next sequence
    self.current_step = 0 # keep track of the current step in the sequence, and also the label
    self.is_first = True
    self.is_last = False

    # Reset and initilize data for the sequence
    data_path, label_path = self.all_paths[self.current_seq_idx[self.current_seq_id]]
    self.event_seq = h5py.File(data_path, "r")["events"] # event under the form: t, x, y, p. All are integers
    self.event_seq = np.asarray(self.event_seq[:], dtype=EVENT_DTYPE)
    self.labels = load_label_file(label_path)
    self.current_event_id = 0 # pointer to the current event in the sequence
    self.current_time = 0 + self.timebin # current timestamp of the sequence in nanoseconds

  def _reset_dataset(self):
    """Reset the whole dataset and start from the beginning

    Returns:
        _type_: _description_
    """
    # pointer to self.current_seq_idx
    self.current_seq_id = -1
    # pointer to self.all_paths (which is already filtered by mode)
    self.current_seq_idx = self.rng.permutation(np.arange(len(self.all_paths)))
    self.is_dataset_first = True
    self.is_dataset_last = False

  @functools.cached_property
  def obs_space(self):
    frame_space = self.event_transform.transform_space
    if self.height != DEFAULT_IMAGE_HEIGHT or self.width != DEFAULT_IMAGE_WIDTH:
      new_shape = (self.height, self.width, frame_space.shape[-1])
      frame_space = Space(frame_space.dtype, new_shape)
    return {
      'frame': frame_space,
      'is_first': Space(dtype=bool),
      'is_last': Space(dtype=bool),
      'is_dataset_first': Space(dtype=bool),
      'is_dataset_last': Space(dtype=bool),
      'label_mask': Space(dtype=bool),  # True if the label is valid
    }

  @functools.cached_property
  def label_space(self):
    return {
      'pupil': Space(dtype=np.int32, shape=(2,), low=0, high=[self.height - 1, self.width - 1]), # y, x
    }

  def get_dataset_info(self) -> Dict[str, int]:
    """Get information about the dataset split.
    
    Returns:
        Dict containing dataset information including total sequences,
        train sequences, eval sequences, and current mode sequences.
    """
    total_sequences = len(self.seq_dirs)
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
    if self.is_dataset_last:
      self._reset_dataset()
    if self.is_last:
      self._reset_seq()

    next_event_id = np.searchsorted(self.event_seq['t'], self.current_time)
    events = self.event_seq[self.current_event_id : next_event_id]
    label = self.labels[self.current_step][:2]
    is_first = self.current_step == 0
    is_last = self.current_step == len(self.labels) - 1
    is_dataset_first = self.current_seq_id == 0
    is_dataset_last = self.current_seq_id == len(self.all_paths) - 1  # Use filtered dataset length
    obs = self._obs(events, label, is_first, is_last, is_dataset_first, is_dataset_last)

    # Update the state
    self.is_first = is_first
    self.is_last = is_last
    self.is_dataset_first = is_dataset_first
    self.is_dataset_last = is_dataset_last
    self.current_event_id = next_event_id
    self.current_time += self.timebin
    self.current_step += 1
    return obs

  def _obs(self, events: np.ndarray, label: np.ndarray, is_first: bool,
      is_last: bool, is_dataset_first: bool, is_dataset_last: bool) -> Dict[str, np.ndarray]:
    frame = self.event_transform(events, original_width=DEFAULT_IMAGE_WIDTH, original_height=DEFAULT_IMAGE_HEIGHT)
    # NOTE: rescale label if needed
    if self.height != DEFAULT_IMAGE_HEIGHT or self.width != DEFAULT_IMAGE_WIDTH:
      label[0] = label[0] / DEFAULT_IMAGE_HEIGHT * self.height
      label[1] = label[1] / DEFAULT_IMAGE_WIDTH * self.width
      label = label.astype(np.int32).clip(0, np.array([self.height - 1, self.width - 1], dtype=np.int32))
    return {
      'frame': frame,
      'pupil': label,
      'is_first': is_first,
      'is_last': is_last,
      'is_dataset_first': is_dataset_first,
      'is_dataset_last': is_dataset_last,
      'label_mask': True
    }


"""
File: threeet.py
Author: Viet Nguyen
Date: 2026-02-10

ThreeET Dataset Loader

This loader handles the ThreeET event-based eye tracking dataset format:
- Event data stored in HDF5 files (.h5)
- Labels stored in text files (label.txt)
- Each sequence is an independent episode
"""

import pathlib
from typing import List, Tuple
import numpy as np
import h5py

from ...utils import Space
from ..loader import Loader
from ..types import EyeEpisodeDataType, EVENT_DTYPE


DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_TIMEBIN = 10000  # 10ms in microseconds


def load_label_file(file_path: str) -> np.ndarray:
  """Load a label.txt file containing (x, y, label) coordinates.

  Args:
    file_path: Path to the label.txt file

  Returns:
    numpy array of shape (N, 3) containing (y, x, label) coordinates
  """
  data = []
  with open(file_path, 'r') as f:
    for line in f:
      # Remove parentheses and split by comma
      line = line.strip().strip('()').split(',')
      # Convert to integers (note: storing as y, x, label)
      x, y, label = map(int, line)
      data.append([y, x, label])
  # Convert to numpy array
  return np.array(data, dtype=np.int32)


class ThreeetLoader(Loader):
  """
  Loader for ThreeET event-based eye tracking dataset.

  Dataset structure:
    dataset_path/
      event_data/
        event_data/
          train/
            {seq_id}/
              {seq_id}.h5     # Event data
              label.txt        # Pupil coordinates

  Each sequence is treated as a separate episode.
  """

  TIMEBIN: int = DEFAULT_TIMEBIN  # 10ms in microseconds

  def __init__(
    self,
    dataset_path: str | pathlib.Path,
    split: str = "train",
    val_ratio: float = 0.2,
    seed: int = 42,
    **kwargs
  ):
    """
    Initialize loader with dataset path and format-specific arguments.

    Args:
      dataset_path: Path to dataset root directory
      split: One of "train", "val", "test". If "test", loads from test/ directory.
             If "train" or "val", loads from train/ directory and splits based on val_ratio.
      val_ratio: Ratio of training data to use for validation (0.0 to 1.0).
                Only used when split is "train" or "val".
      seed: Random seed for reproducible train/val split
    """
    self.dataset_path = pathlib.Path(dataset_path)
    self.split = split
    self.val_ratio = val_ratio
    self.seed = seed

    # Determine which directory to load from
    if split == "test":
      dir_name = "test"
    else:  # train or val
      dir_name = "train"

    # Navigate to the event data directory
    eventdir = self.dataset_path / "event_data" / "event_data" / dir_name
    if not eventdir.exists():
      raise ValueError(f"Event data directory not found: {eventdir}")
    # Collect all sequence directories
    self.seq_dirs: List[pathlib.Path] = sorted([
      seq_dir for seq_dir in eventdir.glob("*")
      if seq_dir.is_dir()
    ])
    if len(self.seq_dirs) == 0:
      raise ValueError(f"No sequence directories found in {eventdir}")
    # Build episode paths (data_path, label_path) for each sequence
    self.episode_paths: List[Tuple[pathlib.Path, pathlib.Path]] = []
    for seq_dir in self.seq_dirs:
      seq_name = seq_dir.name
      data_path = seq_dir / f"{seq_name}.h5"
      label_path = seq_dir / "label.txt"
      # Verify files exist
      if data_path.exists() and label_path.exists():
        self.episode_paths.append((data_path, label_path))
      else:
        print(f"Warning: Missing files for sequence {seq_name}, skipping")
    if len(self.episode_paths) == 0:
      raise ValueError(f"No valid episodes found in {eventdir}")

    # Apply train/val split if needed
    if split in ["train", "val"] and val_ratio > 0.0:
      # Create reproducible split
      rng = np.random.default_rng(seed)
      total_episodes = len(self.episode_paths)
      indices = np.arange(total_episodes)
      rng.shuffle(indices)

      # Calculate split point
      val_size = int(total_episodes * val_ratio)
      train_size = total_episodes - val_size

      if split == "train":
        # Use first train_size episodes for training
        selected_indices = indices[:train_size]
      else:  # val
        # Use last val_size episodes for validation
        selected_indices = indices[train_size:]

      # Filter episode paths based on selected indices
      self.episode_paths = [self.episode_paths[i] for i in sorted(selected_indices)]

    print(f"ThreeET Loader initialized with {len(self.episode_paths)} episodes (split={split})")

  def __len__(self) -> int:
    """
    Return number of episodes in dataset (list-like interface).

    Returns:
      Number of episodes
    """
    return len(self.episode_paths)

  def __getitem__(self, idx: int) -> EyeEpisodeDataType:
    """
    Load episode data using list-like indexing (loader[idx]).

    Args:
      idx: Episode index to load

    Returns:
      EyeEpisodeDataType containing data and labels for the episode
    """
    if idx < 0 or idx >= len(self.episode_paths):
      raise IndexError(f"Episode index {idx} out of range [0, {len(self.episode_paths)})")
    data_path, label_path = self.episode_paths[idx]
    # Load events from HDF5 file
    with h5py.File(data_path, "r") as f:
      # Events are stored with format: t, x, y, p (all integers)
      events_raw = f["events"][:]
    # Convert to structured array with EVENT_DTYPE
    data = np.asarray(events_raw, dtype=EVENT_DTYPE)
    # Load labels from text file
    # Labels format: (y, x, label) where we only need (y, x) for pupil position
    labels = load_label_file(str(label_path))  # Shape: (N, 3)
    # Return episode data
    return EyeEpisodeDataType(
      data=data,
      labels=labels,  # Keep all 3 columns (y, x, label)
      width=DEFAULT_IMAGE_WIDTH,
      height=DEFAULT_IMAGE_HEIGHT,
      timebin=self.TIMEBIN
    )

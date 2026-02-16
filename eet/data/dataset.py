"""
File: dataset.py
Author: Viet Nguyen
Date: 2026-01-11

RLS Dataset module for Robot Learning Systems.

This module provides the core dataset infrastructure for offline learning:
- ShardedDataset: Abstract base for shard-based data access
- RLSDataset: Generic sharded dataset with sequential/single-step sampling
- RLSMixtureDataset: PyTorch IterableDataset for multi-dataset training
"""

import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, List, Dict, Tuple
from omegaconf import open_dict, DictConfig as Config
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from tqdm import tqdm
import torch

from .types import EyeEpisodeDataType, BatchItemDataType
from .loader import Loader
from .processor import BaseProcessor
from .processors.general import GeneralProcessor

def _get_loader_mapping() -> Dict[str, Loader]:
  """Get mapping of dataset names to loader classes."""
  from .loaders.threeet import ThreeetLoader
  # from .loaders.eveye import
  return {
    'threeet': ThreeetLoader,
  }


def build_dataset(config: Config, split: str = "train") -> 'MixtureDataset':
  all_datasets = []
  all_weights = []

  # Get loader mapping with lazy import
  loader_mapping: dict[str, type] = _get_loader_mapping()
  processor: BaseProcessor = GeneralProcessor(config.data.event_repr, config.data.downscale_width, config.data.downscale_height)

  for dataset_name in tqdm(
      config.data.datasets,
      total=len(config.data.datasets),
      desc="Initializing datasets",
  ):
    print(f" - Dataset name: {dataset_name}")
    datasets = []

    # Get loader class for this dataset type
    loader_class: type[Loader] = loader_mapping.get(dataset_name)
    if loader_class is None:
      raise ValueError(f"Unknown dataset type: {dataset_name}. Available types: {list(loader_mapping.keys())}")

    for dataset_path in config.data[dataset_name].dataset_paths:
      print(f"   - Dataset path: {dataset_path}")

      # Instantiate the loader
      loader_kwargs: Config = config.data[dataset_name]
      with open_dict(loader_kwargs):
        _ = loader_kwargs.pop('dataset_paths', None)

      # Add split parameter to loader kwargs
      loader: Loader = loader_class(dataset_path=dataset_path, split=split, **loader_kwargs)

      # Create RLSDataset with the loader
      dataset: 'Dataset' = Dataset(
        loader=loader,
        processor=processor,
        sequence_length=config.batch_length,
        stride=config.data.stride,
        seed=config.seed,
      )
      datasets.append(dataset)

    # compute weights
    dataset_lengths: np.ndarray = np.array([len(dataset) for dataset in datasets]) # n_episodes per dataset
    dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()

    for dataset, relative_length in zip(datasets, dataset_relative_lengths):
      weight = relative_length * config.data[dataset_name].mix_ratio
      all_datasets.append(dataset)
      all_weights.append(weight)

  return MixtureDataset(
    datasets = all_datasets,
    weights = all_weights,
    seed = config.seed,
    training = (split == 'train')
  )

class Dataset:
  """
  Base Dataset with generic logic.

  This class adds sharding capability to any BaseLoader implementation.
  It handles:
  - Sequential or single-step sampling
  - Episode-aware sequence extraction

  Args:
    dataset_path: Path to dataset root directory
    loader: BaseLoader instance for loading episodes
    sequence_length: Number of consecutive timesteps per sequence (1 = single-step)
    stride: Step size between sequence starts (None = sequence_length for non-overlapping)
    seed: Random seed for reproducible sharding
    processor: Optional data processor for transformations
  """
  def __init__(self,
      loader: Loader,
      processor: BaseProcessor,
      sequence_length: int = 32,
      stride: int | None = None,
      seed: int = 42
  ):
    self.dataset_path = loader.dataset_path
    self.loader: Loader = loader
    self.sequence_length = sequence_length
    self.stride = stride if stride is not None else sequence_length
    self.seed = seed
    self.processor = processor
    self.rng = np.random.default_rng(seed)

    self.shuffled_episode_indices: np.ndarray = None
    self.initialize()

  def initialize(self):
    num_episodes: int = len(self.loader) # total number of episodes in the entire dataset
    shuffled_episode_indices: np.ndarray = self.rng.permutation(num_episodes) # (n_episodes,)
    assert len(shuffled_episode_indices) > 0, (
      f"No valid episodes found for dataset {self.dataset_path}"
    )
    # Store shuffled order for episode concatenation during sequence extraction
    self.shuffled_episode_indices = shuffled_episode_indices

  def __len__(self):
    return len(self.loader)

  def get_episode_length(self, idx: int) -> int:
    return self.loader[idx].n_labels

  def extract_episode_data(self, idx: int) -> List[BatchItemDataType]:
    """
    Load and process all sequences in a single episode.

    Splits the episode into sequences of `sequence_length` timesteps with `stride` offset.
    Each sequence is a BatchItemDataType containing events and labels for consecutive timebins.

    Args:
      idx: the index of the episode

    Returns:
      List of processed sequences (BatchItemDataType instances)
    """
    episode_data: EyeEpisodeDataType = self.loader[idx]
    datapoints: List[BatchItemDataType] = []

    # Get episode metadata
    total_labels = episode_data.n_labels
    timebin = episode_data.timebin
    width = episode_data.width
    height = episode_data.height
    events = episode_data.events
    labels = episode_data.labels

    # Pre-process: Split all events into timebins once (optimization to avoid redundant searchsorted)
    all_timebin_events: List[np.ndarray] = []

    # Vectorized approach: compute all timebin boundaries at once
    timebin_boundaries = np.arange(total_labels + 1) * timebin  # [0, timebin, 2*timebin, ...]
    split_indices = np.searchsorted(events['t'], timebin_boundaries, side='left')

    # Split events into timebins using the precomputed indices
    for i in range(total_labels):
      start_idx = split_indices[i]
      end_idx = split_indices[i + 1]
      timebin_events = events[start_idx:end_idx]
      all_timebin_events.append(timebin_events)

    # Extract sequences with stride by slicing pre-processed timebins
    for start_idx in range(0, total_labels - self.sequence_length + 1, self.stride):
      end_idx = start_idx + self.sequence_length

      # Slice pre-processed events for this sequence
      sequence_events = all_timebin_events[start_idx:end_idx]

      # Slice labels for this sequence
      sequence_labels = [labels[i] for i in range(start_idx, end_idx)]

      # Create BatchItemDataType
      batch_item = BatchItemDataType(
        events=sequence_events,
        labels=sequence_labels,
        width=width,
        height=height,
        timebin=timebin,
        batch_length=self.sequence_length
      )
      # The processor must exist to process the batch data
      self.processor(batch_item)

      datapoints.append(batch_item)

    return datapoints


class MixtureDataset(IterableDataset):
  """
  PyTorch IterableDataset that combines multiple datasets.

  This is the final dataset class used in the training pipeline. It provides:
  1. Weighted sampling across multiple datasets
  2. Intelligent shard sampling accounting for dataset sizes
  3. Distributed training support (multi-worker, multi-GPU)
  4. Epoch management

  The sampling strategy ensures datasets are sampled proportionally to their
  weights while accounting for differences in shard sizes, preventing bias.

  Args:
    datasets: List of RLSDataset instances to combine
    weights: Mixing weights for each dataset (will be normalized)
    seed: Random seed for reproducible sampling
    training: Whether in training mode (affects sampling strategy)

  Example:
    >>> mixture = MixtureDataset(
    ...     datasets=[dataset1, dataset2],
    ...     weights=[0.7, 0.3],
    ... )
    >>> dataloader = DataLoader(mixture, batch_size=None, num_workers=4)
    >>> for batch in dataloader:
    ...     loss = model(batch)
    ...     optimizer.step()
  """

  def __init__(
    self,
    datasets: List[Dataset],
    weights: List[float],
    seed: int = 42,
    training: bool = True
  ):
    self.datasets = datasets
    self.weights = np.array(weights, dtype=np.float64)
    self.weights = self.weights / self.weights.sum()  # Normalize
    self.seed = seed
    self.training = training
    self.refresh_on_epoch = True  # Refresh datasets at epoch start

    # Compute episode sampling probabilities
    self._compute_sampling_weights()

    # Initialize caching system (created in __iter__ to avoid pickling issues)
    self.curr_episode: List[BatchItemDataType] | None = None
    self._executor: ThreadPoolExecutor | None = None
    self._cache_job: Future | None = None

  def _compute_sampling_weights(self):
    """
    Compute sampling probabilities for each dataset.

    Weights are normalized by total samples to prevent bias toward
    datasets with smaller episodes.
    """
    # Total samples per dataset
    total_samples = np.array([
      sum(ds.get_episode_length(i) for i in range(len(ds)))
      for ds in self.datasets
    ])

    # Normalize weights by samples (prevent bias)
    self.sampling_probs = self.weights * total_samples
    self.sampling_probs = self.sampling_probs / self.sampling_probs.sum()

    print(f"Dataset sampling probabilities: {self.sampling_probs}")

  def __iter__(self):
    """
    Iterate over the mixture dataset with background episode caching.

    Implements an efficient iteration strategy:
    1. Start background thread pool for caching
    2. Determine worker-specific episode schedule
    3. For each episode: wait for cache, start caching next, yield current
    4. Shuffle sequences within each episode for additional randomization
    5. Clean up cached episodes to free memory
    """
    # Start background thread pool for episode caching
    self._executor = ThreadPoolExecutor(max_workers=1)

    # Determine which episodes this worker handles
    worker_info = get_worker_info()
    if worker_info is None:
      worker_id = 0
      num_workers = 1
    else:
      worker_id = worker_info.id
      num_workers = worker_info.num_workers

    # Create worker-specific RNG
    rng = np.random.default_rng(self.seed + worker_id)

    # Generate episode schedule for this worker
    episode_schedule = []

    # Calculate total episodes across all datasets
    total_episodes = sum(len(ds) for ds in self.datasets)

    for epoch_episode_idx in range(total_episodes):
      # Only include episodes assigned to this worker
      if epoch_episode_idx % num_workers != worker_id:
        continue

      # Sample dataset according to weights
      dataset_idx = rng.choice(len(self.datasets), p=self.sampling_probs)
      dataset = self.datasets[dataset_idx]

      # Sample random episode from chosen dataset
      episode_idx = rng.integers(0, len(dataset))

      episode_schedule.append((dataset_idx, episode_idx))

    # Start caching the first episode
    if len(episode_schedule) > 0:
      self.cache_next_episode(episode_schedule, 0)

    # Iterate through all scheduled episodes
    for i in range(len(episode_schedule)):
      # Wait for background caching to complete
      self.finish_cache_episode()

      # Start caching next episode immediately (if not last)
      if i + 1 < len(episode_schedule):
        self.cache_next_episode(episode_schedule, i + 1)

      # Yield shuffled samples from current episode
      assert self.curr_episode is not None
      if self.training:
        indices_in_episode = np.arange(len(self.curr_episode))
        rng.shuffle(indices_in_episode)
        for idx in indices_in_episode:
          yield self.curr_episode[idx].processed
      else:
        for sample in self.curr_episode:
          yield sample.processed

      # Clean up cached episode to free memory
      self.delete_cached_episode()

    # Shutdown executor after iteration completes
    if self._executor is not None:
      self._executor.shutdown(wait=True)
      self._executor = None

  def cache_next_episode(self, episode_schedule: List[Tuple[int, int]], schedule_idx: int):
    """
    Start background caching of an episode using ThreadPoolExecutor.

    Args:
      episode_schedule: List of (dataset_idx, episode_idx) tuples
      schedule_idx: Index in the schedule to cache
    """
    assert self._executor is not None
    dataset_idx, episode_idx = episode_schedule[schedule_idx]
    # Submit background loading job
    self._cache_job = self._executor.submit(
      self.datasets[dataset_idx].extract_episode_data, episode_idx
    )

  def finish_cache_episode(self):
    """Wait for the background caching job to complete and retrieve the episode."""
    assert self._cache_job is not None
    self.curr_episode = self._cache_job.result()
    self._cache_job = None

  def delete_cached_episode(self):
    """Delete the current cached episode to free memory."""
    del self.curr_episode
    self.curr_episode = None

  def __len__(self):
    """
    Return total number of samples across all episodes.

    Note: This is approximate for IterableDataset and not used by DataLoader.
    """
    # return sum(
    #   sum(ds.get_episode_length(i) for i in range(len(ds)))
    #   for ds in self.datasets
    # )
    # the number of batch item can be computed by getting the total number of timesteps in all episodes
    #   then work through the sequence length (with stride logic), then divide by the batch size
    # one episode will contain some timesteps, one batch item will contain a
    #   fix sequence length, and will be offset by stride. Finally to get the number of
    #   batch, we can divide the total by the batch size
    """
    Return total number of sequences (batch items) across all episodes.

    For each episode with N timesteps:
      - Number of sequences = (N - sequence_length) // stride + 1 (if N >= sequence_length)

    The DataLoader will then divide this by batch_size to get number of batches.

    Note: This is approximate for IterableDataset with multiple workers, as each
    worker only processes a subset of episodes.
    """
    total_sequences = 0
    for ds in self.datasets:
      for episode_idx in range(len(ds)):
        episode_length = ds.get_episode_length(episode_idx)
        # Calculate number of sequences this episode produces
        if episode_length >= ds.sequence_length:
          num_sequences = (episode_length - ds.sequence_length) // ds.stride + 1
          total_sequences += num_sequences
    return total_sequences


def build_dataloader(
    dataset: MixtureDataset,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    persistent_workers: bool = True
):
  from .collator import GeneralCollator
  assert isinstance(dataset, MixtureDataset), "Expected dataset to be a MixtureDataset instance"
  # collator = GeneralCollator()
  return DataLoader(
    dataset,
    batch_size=batch_size,
    # collate_fn=collator,
    num_workers=num_workers,
    drop_last=True,
    prefetch_factor=prefetch_factor,  # Prefetch 4 batches per worker
    persistent_workers=persistent_workers  # Keep workers alive between epochs
  )



# %%

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from eet.data.loader import Loader
from eet.data.dataset import Dataset, MixtureDataset, build_dataloader
from eet.data.loaders.threeet import ThreeetLoader
from eet.data.processors.general import GeneralProcessor

from eet.data.types import EyeEpisodeDataType, BatchItemDataType

# NOTE: replace with your path
threeet_path = pathlib.Path('/home/vn1747/tsxai/ai-pipeline/experimental/eet/data/event-based-eye-tracking-ais2024')

loader = ThreeetLoader(threeet_path, split="train", val_ratio=0.2, seed=42)

episode: EyeEpisodeDataType = loader[4]

print(len(loader))

# %%

loader[3].events['t']


# %%

dataset = Dataset(loader, GeneralProcessor("binarep", 80, 60), sequence_length=128, stride=32)

# %%

print(len(dataset))

# %%

print(dataset.get_episode_length(7))

# %%

data: List[BatchItemDataType] = dataset.extract_episode_data(3)

# %%

mixture: MixtureDataset = MixtureDataset([dataset], [1.0])

# %%

it = iter(mixture)

# %%

batch = next(it)


# %%

batch.keys()

# %%

# Draw the event frame with the label (gaze point)

for t in range(batch['event'].shape[0]):
  clear_output(wait=True)

  frame = batch['event'][t]  # Shape: (H, W, C)
  label = batch['label'][t]  # Shape: (2,) - [x, y] where x is width, y is height

  # Create a figure and axis
  fig, ax = plt.subplots(1, 1, figsize=(10, 8))

  # Display the frame
  ax.imshow(frame)

  # Extract x and y coordinates from the label
  y, x = label[0], label[1]

  # Draw a marker at the gaze point
  ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3, label='Gaze Point')
  ax.plot(x, y, 'yo', markersize=10, fillstyle='none', markeredgewidth=2)

  # Add a title and legend
  ax.set_title(f'Event Frame with Gaze Point at ({x:.1f}, {y:.1f})')
  ax.legend()

  plt.show()


# %%


dataloader = build_dataloader(mixture, batch_size=32, num_workers=4)



# %%


len(dataloader) # 808

# %%

it = iter(dataloader)

# %%

batch = next(it)

# %%

batch['event'].shape

# %%

# 20 batch
for i, batch in enumerate(dataloader):
  print(f"processed batch {i}")

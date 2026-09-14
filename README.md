# Event Based Eye Tracking Pipeline

A library for running and evaluating different event-based eye tracking algorithms.

## Project Structure

```
.
├── data/                # Directory for datasets
├── configs/             # Configuration files for experiments (we use hydra config)
├── eet/                 # Main package
│ ├── agents/            # Agent implementations
│ ├── data/              # Data handling modules
│ ├── networks/          # Neural network implementations
│ ├── optim/             # Optimization algorithms
│ ├── utils/             # Utility functions
│ └── trainer.py         # Trainer module
└── scripts/             # Utility scripts
    └── train.py         # Training script
```

## Setup

### Installation

```bash
conda create -n eet python=3.10
conda activate eet
bash scripts/install.sh
```

## Usage

* Train on 3ET dataset:

```bash

# NOTE: we can add multiple datasets by separating them with comma, e.g. data.datasets=[threeet,another_dataset]
python scripts/train.py data.datasets=[threeet]

# AISSM agent
python scripts/train.py agent.name=aissm data.datasets=[threeet] agent.compile=True


# To train and test on different resolution, just specify the resolution
#   in the config, e.g., `data.processor_kwargs.general.height=320 data.processor_kwargs.general.width=320`
#   For example:
python scripts/train.py agent.name=aissm data.datasets=[threeet] agent.compile=True data.processor_kwargs.general.height=320 data.processor_kwargs.general.width=320 batch_size=4 batch_length=16

# Some other examples
python scripts/train.py agent.name=aissm data.datasets=[threeet] agent.compile=True batch_length=16 batch_size=16 expname=aissm

```



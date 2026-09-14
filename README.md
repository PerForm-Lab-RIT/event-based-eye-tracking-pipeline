# Event Based Eye Tracking Pipeline

A library for running and evaluating different event-based eye tracking algorithms.

This is the official implementation for the paper: [Enhancing Eye Feature Estimation from Event Data Streams through Adaptive Inference State Space Modeling](https://doi.org/10.1145/3797246.3803041) (ETRA '26).


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


## Other Information

### Contributors

Special thanks to the following contributors:

* [Viet Dung Nguyen](https://vietdung.me)
* [Mobina Ghorbaninejad](https://github.com/MobinaGhorbaninejad)

### Citation

```bibtex
@inproceedings{Nguyen2026AISSM,
  author = {Nguyen, Viet Dung and Ghorbaninejad, Mobina and Ma, Chengyi and Bailey, Reynold and Diaz, Gabriel and Fix, Alexander and Suess, Ryan J and Ororbia, Alexander},
  title = {Enhancing Eye Feature Estimation from Event Data Streams through Adaptive Inference State Space Modeling},
  year = {2026},
  isbn = {9798400725197},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3797246.3803041},
  doi = {10.1145/3797246.3803041},
  abstract = {Eye feature extraction from event-based data streams can be performed efficiently and with low energy consumption, offering great utility to real-world eye tracking pipelines. However, few eye feature extractors are designed to handle sudden changes in event density caused by the changes between gaze behaviors that vary in their kinematics, leading to degraded prediction performance. In this work, we address this problem by introducing the adaptive inference state space model (AISSM), a novel architecture for feature extraction that is capable of dynamically adjusting the relative weight placed on current versus recent information. This relative weighting is determined via estimates of the signal-to-noise ratio and event density produced by a complementary dynamic confidence network. Lastly, we craft and evaluate a novel learning technique that improves training efficiency. Experimental results demonstrate that the AISSM system outperforms state-of-the-art models for event-based eye feature extraction.},
  booktitle = {Proceedings of the 2026 Symposium on Eye Tracking Research and Applications},
  articleno = {8},
  numpages = {9},
  location = {},
  series = {ETRA '26}
}
```




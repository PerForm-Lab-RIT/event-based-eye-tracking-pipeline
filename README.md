# Event Based Eye Tracking Pipeline

A library for running and evaluating different event-based eye tracking algorithms.

## Project Structure

```
.
├── data/                # Directory for datasets
├── docs/                # Documentation files
├── eet/                 # Main package
│ ├── common/            # Common utilities and functions
│ ├── data/              # Data handling modules
│ ├── models/            # Eye tracking algorithm models
│ ├── nn/                # Neural network implementations
│ ├── configs.yaml       # Configuration file
│ └── __init__.py        # Package initialization
└── scripts/             # Utility scripts
    ├── eval.py          # Evaluation script
    └── train.py         # Training script
```

## Getting Started

### Installation

```bash
conda create -n eet python=3.12
conda activate eet
bash install.sh
```

### Usage

```bash
# Train the model
python scripts/train.py <some more arguments here>

# Evaluate the model
python scripts/eval.py <some more arguments here>
```


## Documentation

[Link to documentation will go here]

## License

[License information will go here]




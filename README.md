# Event Based Eye Tracking Pipeline

A library for running and evaluating different event-based eye tracking algorithms.

## Project Structure

```
.
├── data/                # Directory for datasets
├── docs/                # Documentation files
├── eet/                 # Main package
│ ├── agents/            # Eye tracking algorithm implementations
│ ├── envs/              # Environment and data handling modules
│ ├── modeling/          # Neural network model components
│ ├── utils/             # Utility functions
│ ├── configs.yaml       # Configuration file
│ └── __init__.py        # Package initialization
├── lib/                 # Core library components
│ ├── agent/             # Agent base classes
│ ├── common/            # Common utilities and functions
│ ├── cv/                # Computer vision utilities
│ ├── envs/              # Environment base classes
│ ├── nn/                # Neural network implementations
│ ├── replay/            # Replay buffer implementations
│ └── utils/             # Utility functions
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




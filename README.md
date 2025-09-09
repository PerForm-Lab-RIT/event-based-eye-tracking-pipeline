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

### About the CLI and config file

* To train an agent you might just want to run: `python scripts/train.py --configs <primitive1> <primitive2> ... --<key1> <value1> --<key2> <value2> ...` (make sure you check the file to see what it actually does)

* For example, to train a CNN agent on the 3ET+ dataset, you can run:

```bash
python scripts/train.py --logroot logs --expname 3et-cnn-experiment01 --agent cnn --task threeet --.*\.mults 2,3,4,4 --.*\.depth 4 --.*\.units 128 --.*\.hidden 64 --batch_size 64 --batch_length 32
```

where `--logroot` specifies the root directory for logs, `--expname` specifies the experiment name, `--agent` specifies the agent type, and `--task` specifies the task/environment/dataset. `--.*\.mults 2,3,4,4 --.*\.depth 4 --.*\.units 128 --.*\.hidden 64` specifies the agent network configuration. Other configurations such as `--batch_size` and `--batch_length` can also be specified.

* You can also run with config primitives to shorten the command line: `python scripts/train.py --configs <primitive1> <primitive2> ...`. For example:

```bash
python scripts/train.py --configs 3et_160x120 cnn_500k --logroot logs --expname 3et-cnn-experiment01
```
This also works because the config primitive name `3et_160x120` and `cnn_500k` has already been defined in the config file `configs.yaml`

* Structure of the config file. First, the yaml config file contanins multiple key-value pairs, where the first layer of the keys are `defaults` and primitive names. The loader will load the `defaults` first, the based on the provided arguments on the command line you provide, it will merge the config primitives with the defaults (if any). It will also merge the argument you pass in with the defaults to compose a final config (similar to hydra config but a lighter version).

* Note: the order matter: `--cnn_500k --encoder.units 4` will results in the final encoder units being `4` as it is passed later in the command line. The later ones will override the earlier ones if they are the same key.

* Note: the argument you can pass can be searched, implemented, and examined in the config file `configs.yaml`, the `defaults` section. The config primitive can be searched, implemented, and examined in the config file `configs.yaml` for sections other than the `defaults` section.

* We also accept regex argument: `--.*\.foo bar` means that for all keys in all level that matches the regex `.*\.foo`, its value will be set to `bar`. This is useful for setting multiple keys at once. For example, `--.*\.mults 2,3,4,4` will set all keys that match the regex `.*\.mults` to the array value `[2, 3, 4, 4]`.


## License

[License information will go here]




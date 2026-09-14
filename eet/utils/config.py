import os
import pathlib
from omegaconf import DictConfig as Config, OmegaConf
import yaml
import pprint

def preprocess_config(config: Config) -> Config:
  """Preprocess the config object received from hydra

  Args:
      config (Config): _description_

  Returns:
      Config: _description_
  """
  # check if expname is empty, if so, fill with timestamp
  os.makedirs(config.logroot, exist_ok=True)

  # Resolve interpolations (e.g., cast logdir to pathlib.Path)
  config.logdir = pathlib.Path(config.logdir)
  OmegaConf.resolve(config)

  # Create logging folder
  os.makedirs(config.logdir, exist_ok=True)

  # print config and dump it to a composed yaml file
  dict_config: dict = OmegaConf.to_container(config)
  pprint.PrettyPrinter(indent=4).pprint(dict_config)
  dump = os.path.join(config.logdir, 'config.yaml')
  with open(dump, 'w') as f:
    yaml.dump(dict_config, f)

  return config
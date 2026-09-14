"""
File: main.py
Author: Viet Nguyen
Date: 2025-12-08

Description: Main file
"""

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import os
import multiprocessing as mp
import pprint
import importlib
from omegaconf import DictConfig as Config, OmegaConf
import yaml
import atexit
import hydra
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

import torch

from eet.utils import setup_console_log, preprocess_config
from eet.data import build_dataset, build_dataloader
from eet.agents import build_agent
from eet.trainer import Trainer
from eet.utils.logger import Logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: Config):
  """This function receive the config object from hydra, and start the distributed processes
    Note, for each device, we will start a process

  Args:
      config (Config): _description_
  """
  # process config, replace all unfilled config with actual vars
  config: Config = preprocess_config(config)

  # Mirror stdout/stderr to a file under logdir while keeping console output.
  setup_console_log(config.logdir, filename="console.log")
  atexit.register(setup_console_log, config.logdir, filename="console.log")

  # build datasets and dataloaders
  train_dataset, input_space, label_space = build_dataset(config, split="train")
  val_dataset, _, _ = build_dataset(config, split="val")
  train_loader = build_dataloader(
    train_dataset,
    batch_size=int(config.batch_size),
    num_workers=config.data.num_workers,
    prefetch_factor=config.data.prefetch_factor,
    persistent_workers=config.data.persistent_workers,
  )
  val_loader = build_dataloader(
    val_dataset,
    batch_size=int(config.batch_size),
    num_workers=config.data.num_workers,
    prefetch_factor=config.data.prefetch_factor,
    persistent_workers=config.data.persistent_workers,
  )

  agent = build_agent(config, input_space, label_space)

  logdir = pathlib.Path(config.logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  tb_logger = Logger(logdir=logdir)
  tb_logger.log_hydra_config(config)

  trainer = Trainer(
    config=config.trainer,
    train_loader=train_loader,
    val_loader=val_loader,
    agent = agent,
    logger = tb_logger,
  )
  trainer.run()



if __name__ == '__main__':
  """This is the main call to the main program
  """
  main()
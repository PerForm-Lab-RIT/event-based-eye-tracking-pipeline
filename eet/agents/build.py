
from omegaconf import OmegaConf, DictConfig as Config
import torch.nn as nn

from ..data.dataset import IterableDataset
from ..utils import Space
from ..agents import BaseAgent

from .aissm import AISSMAgent

ALL_AGENTS = {
  "aissm": AISSMAgent,
}

def build_agent(config: Config, input_space: Space, label_space: Space) -> nn.Module:
  agent: BaseAgent = ALL_AGENTS[config.agent.name](config.agent, input_space, label_space)
  agent.to(config.device)
  return agent



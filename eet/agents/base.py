from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
from torch import nn
from omegaconf import DictConfig as Config
import functools
# from calflops import calculate_flops

from ..optim import OptimizerEngine
from ..utils import Space, print

# Automatically call post_init() after __init__() to initialize optimizer and compile grad if needed
class BaseAgentMeta(type):
  def __call__(cls, *args, **kwargs):
    new_obj = type.__call__(cls, *args, **kwargs)
    new_obj.post_init()
    return new_obj

class BaseAgent(nn.Module, metaclass=BaseAgentMeta):
  """Base class for all agents, defines the basic interface and common functionalities such as optimizer setup, save/load, etc. 
  Each agent should implement the report, and loss functions. The grad function is implemented in the base class to perform
  the backward step, but can be overridden if the agent needs a different training loop.

  Main functions to implement:
    - report: return the metrics to log without updating the model, used for validation. If you never call report(),
      then you can just return an empty dict, or not implementing it at all
    - loss: the loss function can optionally return auxiliary outputs from the third output onward.
      If you ever train the agent using `update()`, then you should implement the loss function

  NOTE: It is important to set the all_modules dict for the optimizer / save / load functionalities to work
  NOTE: To train the agent for one step, call `update()` function
  """
  def __init__(self, config: Config, input_space: Space, label_space: Space):
    super().__init__()
    self.config = config
    self.input_space = input_space
    self.label_space = label_space

    self.device = torch.device(config.device)
    self.all_modules = nn.ModuleDict()  # store all submodules in a dict for easy access when building optimizers
    print(f'  {"Input Space":<16} {input_space}')
    print(f'  {"Label Space":<16} {label_space}')

  def post_init(self):
    self.optimizer = OptimizerEngine(
			sum([list(m.parameters()) for m in self.all_modules.values()], []),
			**self.config.opt,
		)
    self.train()
    if self.config.compile:
      self.grad = torch.compile(self.grad, mode='reduce-overhead')

  @functools.cached_property
  def trainable_n_params(self) -> int:
    return sum(p.numel() for m in self.all_modules.values() for p in m.parameters() if p.requires_grad)

  @functools.cached_property
  def n_params(self) -> int:
    return sum(p.numel() for m in self.all_modules.values() for p in m.parameters())

  @functools.cached_property
  def flops(self) -> float | str:
    raise NotImplementedError("flops property not implemented for this agent")

  # IMPLEMENT THIS IN THE CHILD CLASS
  def report(self, inputs: torch.Tensor, label: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
    """Given the input tensor and label tensor, return a dict of metrics to log"""
    raise NotImplementedError("(inputs, label) -> metrics")

  # IMPLEMENT THIS IN THE CHILD CLASS
  def loss(self, inputs: torch.Tensor, label: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
    """Calculate gradients and return metrics to log"""
    raise NotImplementedError("(inputs, label) -> loss, metrics, *aux_outputs")

  # IMPLEMENT THIS IN THE CHILD CLASS
  def forward(self, inputs: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
    """Calculate gradients and return metrics to log"""
    raise NotImplementedError("(inputs) -> outputs")

  # IMPLEMENT THIS IN THE CHILD CLASS
  def decode_outputs(self, outputs: Any) -> torch.Tensor:
    """Return the prediction in pixel space, given the raw model output from forward()"""
    raise NotImplementedError("(outputs) -> predictions_px")


  #### The following functions have default implementations but can be overridden if needed ####

  def update(self, inputs: torch.Tensor, label: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
    """Train one step given the batch data

    Args:
        inputs (torch.Tensor): input tensor of shape (B, T, C, H, W)
        label (torch.Tensor): label tensor of shape (B, T, 2)

    Returns:
        Dict[str, torch.Tensor]: _description_
    """
    if self.config.compile:
      torch.compiler.cudagraph_mark_step_begin()
    with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
      metrics = self.grad(inputs, label, *args, **kwargs)
    opt_metrics = self.optimizer()
    metrics.update(opt_metrics)
    return metrics

  def grad(self, inputs: torch.Tensor, label: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
    """Perform one full gradient step (including loss computation and gradient backward step)

    Args:
        inputs (torch.Tensor): input tensor of shape (B, T, C, H, W)
        label (torch.Tensor): label tensor of shape (B, T, 2)

    Returns:
        Dict[str, torch.Tensor]: dictionary containing the loss and metrics
    """
    outs = self.loss(inputs, label, *args, **kwargs)
    # the loss function can optionally return auxiliary outputs from the third output onward
    loss, metrics = outs[0], outs[1]
    self.optimizer.scaler.scale(loss).backward()
    return metrics

  def save(self) -> Dict[str, Any]:
    return {
			**{k: m.state_dict() for k, m in self.all_modules.items()},
			"optimizer": self.optimizer.save()
		}

  def load(self, state: Dict[str, Any] | None = None) -> None:
    if state is None:
      return
    for k, m in self.all_modules.items():
      m.load_state_dict(state[k])
    self.optimizer.load(state["optimizer"])




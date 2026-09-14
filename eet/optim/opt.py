import torch
from torch import nn
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR

from .laprop import LaProp
from .agc import clip_grad_agc_


def compute_rms(tensors):
  """Compute the root mean square (RMS) of a list of tensors."""
  valid = [t.view(-1) for t in tensors if t is not None]
  if not valid:
    return torch.tensor(0.0)
  flattened = torch.cat(valid)
  if len(flattened) == 0:
    return torch.tensor(0.0)
  return torch.linalg.norm(flattened, ord=2) / (flattened.numel() ** 0.5)


def compute_global_norm(tensors):
  """Compute the global norm (L2 norm) across a list of tensors."""
  valid = [t.view(-1) for t in tensors if t is not None]
  if not valid:
    return torch.tensor(0.0)
  flattened = torch.cat(valid)
  if len(flattened) == 0:
    return torch.tensor(0.0)
  return torch.linalg.norm(flattened, ord=2)


class OptimizerEngine:
  def __init__(self, parameters, lr, beta1=0.9, beta2=0.999,
      eps=1e-20, weight_decay: float = 0.0, clip: float = 0.0,
      agc: float = 0.3, pmin: float = 1e-3,
      optimizer = 'laprop', warmup: int = 1000, log_grads=True):

    # Parameters to optimize
    self.parameters = list(parameters)

    # log gradients
    self._log_grads = log_grads

    # Optimizer initialization
    if optimizer == 'adamw':
      self._optimizer = AdamW(self.parameters, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    elif optimizer == 'laprop':
      self._optimizer = LaProp(
        self.parameters,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
      )
    else:
      raise NotImplementedError(f"Optimizer {optimizer} is not implemented")

    # Gradient clipping by global norm
    def _clip_by_norm(params) -> None:
      torch.nn.utils.clip_grad_norm_(params, float(clip))
    self._clip_by_norm = _clip_by_norm if clip > 0 else None

    # Adaptive gradient clipping
    def _agc(params) -> None:
      clip_grad_agc_(params, float(agc), float(pmin), foreach=True)
    self._agc = _agc if agc > 0 else None

    # Gradient scaler for mixed precision training
    self.scaler = GradScaler()

    # Lambda LR scheduler for learning rate warmup
    def lr_lambda(step) -> float:
      if warmup:
        return min(1.0, (step + 1) / warmup)
      return 1.0
    self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)


  def __call__(self) -> dict:
    """Before calling this, you should compute the gradient using this pattern:
    ```
    self.optimizer = OptimizerEngine(...)
    self.device = torch.device('cuda')

    ... Some code ...

    with autocast(device_type=self.device.type, dtype=torch.float16):
      loss = compute_loss(...)
      self.optimizer.scaler.scale(loss).backward()
    ```

    Returns:
        dict: _description_
    """
    metrics = {}
    self.scaler.unscale_(self._optimizer) # unscale grads in params
    if self._log_grads:
      old_params = [p.data.clone().detach() for p in self.parameters]
      grads = [p.grad for p in self.parameters if p.grad is not None]  # log grads before clipping
      grad_norm = compute_global_norm(grads)
      grad_rms = compute_rms(grads)
      metrics["opt/grad_norm"] = grad_norm
      metrics["opt/grad_rms"] = grad_rms
    if self._clip_by_norm: # clip gradients by global norm
      self._clip_by_norm(self.parameters)
    if self._agc: # adaptive gradient clipping
      self._agc(self.parameters)
    scale_before = self.scaler.get_scale()
    self.scaler.step(self._optimizer) # update parameters (may be skipped on overflow)
    self.scaler.update() # adjust the scale for next iteration
    scale_after = self.scaler.get_scale()
    if scale_after >= scale_before:
      self._scheduler.step() # only advance LR schedule when optimizer step happened
    self._optimizer.zero_grad(set_to_none=True) # zero out gradients after update
    metrics["opt/lr"] = self._scheduler.get_last_lr()[0]
    metrics["opt/grad_scale"] = self.scaler.get_scale()
    if self._log_grads:
      updates = [(new - old) for (new, old) in zip(self.parameters, old_params)]
      update_rms = compute_rms(updates)
      params_rms = compute_rms(self.parameters)
      metrics["opt/param_rms"] = params_rms
      metrics["opt/update_rms"] = update_rms
    return metrics


  def save(self) -> dict:
    """Save the state of the optimizer engine, including the optimizer and scaler states."""
    return {
      "optimizer": self._optimizer.state_dict(),
      "scheduler": self._scheduler.state_dict(),
      "scaler": self.scaler.state_dict(),
    }

  def load(self, state: dict) -> None:
    """Load the state of the optimizer engine from a saved state."""
    self._optimizer.load_state_dict(state["optimizer"])
    self._scheduler.load_state_dict(state["scheduler"])
    self.scaler.load_state_dict(state["scaler"])

"""
File: optimizers.py
Author: Viet Nguyen
Date: 2025-03-21

Description: Heavily adapted some modules from Danijar Hafner's implementation
"""

import math

import jax
import jax.numpy as jnp
import optax
import re

from . import internal
from .. import nn as nets
from . import ninjax as nj
from . import functional as F

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


class Optimizer(nj.Module):

  # optimizer config
  lr: float = 4e-5
  agc: float = 0.3
  eps: float = 1e-20
  beta1: float = 0.9
  beta2: float = 0.999
  nesterov: bool = False
  wd: float = 0.0
  wdregex: str = r'/kernel$'
  schedule: str = 'const'
  warmup: int = 1000
  anneal: int = 0

  summary_depth: int = 2

  def __init__(self, modules):
    """
    Initialize the optimizer.

    Args:
      modules (List[nn.Module]): List of modules to optimize
      lr (float, optional): Learning rate. Defaults to 4e-5. If lr is a number, it is interpreted as a learning rate.
      agc (float, optional): Gradient clipping. Defaults to 0.3. If agc is a number, it is interpreted as a gradient clipping.
      eps (float, optional): Epsilon. Defaults to 1e-20. If eps is a number, it is interpreted as a epsilon.
      beta1 (float, optional): Beta 1. Defaults to 0.9. If beta1 is a number, it is interpreted as a beta 1.
      beta2 (float, optional): Beta 2. Defaults to 0.999. If beta2 is a number, it is interpreted as a beta 2.
      nesterov (bool, optional): Nesterov. Defaults to False. If nesterov is True, the optimizer uses Nesterov momentum.
      wd (float, optional): Weight decay. Defaults to 0.0. If wd is a number, it is interpreted as a weight decay.
      wdregex (str, optional): Weight decay regex. Defaults to r'/kernel$'. If wdregex is a number, it is interpreted as a regex.
      schedule (str, optional): Schedule. Defaults to 'const'. If schedule is 'const', the learning rate is constant.
      warmup (int, optional): number of warmup steps. Defaults to 1000. If warmup is 0, the schedule is constant.
      anneal (int, optional): number of annealing steps. Defaults to 0. If anneal is 0, the schedule is constant.

    Raises:
        NotImplementedError: If the schedule is not implemented
    """
    modules = modules if isinstance(modules, (list, tuple)) else (modules,)
    self.modules = modules

    # Building optax chain ---------------------
    chain = []
    chain.append(clip_by_agc(self.agc))
    chain.append(scale_by_rms(self.beta2, self.eps))
    chain.append(scale_by_momentum(self.beta1, self.nesterov))
    if self.wd:
      assert not self.wdregex[0].isnumeric(), self.wdregex
      pattern = re.compile(self.wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(self.wd, wdmask))
    assert self.anneal > 0 or self.schedule == 'const'
    if self.schedule == 'const':
      sched = optax.constant_schedule(self.lr)
    elif self.schedule == 'linear':
      sched = optax.linear_schedule(self.lr, 0.1 * self.lr, self.anneal - self.warmup)
    elif self.schedule == 'cosine':
      sched = optax.cosine_decay_schedule(self.lr, self.anneal - self.warmup, 0.1 * self.lr)
    else:
      raise NotImplementedError(self.schedule)
    if self.warmup:
      ramp = optax.linear_schedule(0.0, self.lr, self.warmup)
      sched = optax.join_schedules([ramp, sched], [self.warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    self.opt: optax.GradientTransformationExtraArgs = optax.chain(*chain)
    # ----------------------------------------

    # Other setup and chain modification
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (nets.COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')

  def __call__(self, lossfn, *args, has_aux=False, **kwargs):
    metrics = {}

    def lossfn2(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux

    loss, params, grads, aux = nj.grad(
        lossfn2, self.modules, has_aux=True)(*args, **kwargs)
    if self.scaling:
      loss *= 1 / self.grad_scale.read()

    counts = {k: math.prod(v.shape) for k, v in params.items()}
    if nj.creating():
      print(self._summarize_params(counts, self.summary_depth))

    axes = internal.get_data_axes()
    if axes:
      grads = jax.tree.map(lambda x: jax.lax.pmean(x, axes), grads)

    if self.scaling:
      invscale = 1 / self.grad_scale.read()
      grads = jax.tree.map(lambda x: x * invscale, grads)

    state = self.sub('state', nj.Tree, self.opt.init, params)
    updates, new_state = self.opt.update(grads, state.read(), params)
    nj.context().update(optax.apply_updates(params, updates))
    state.write(new_state)
    grad_norm = optax.global_norm(grads)
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + i32(jnp.isfinite(grad_norm)))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = f32(~jnp.isfinite(grad_norm))
    else:
      self.step.write(self.step.read() + 1)
    metrics['loss'] = loss.mean()
    metrics['updates'] = self.step.read()
    metrics['grad_norm'] = grad_norm
    metrics['grad_rms'] = F.rms(grads)
    metrics['update_rms'] = F.rms(updates)
    param_list = []
    for module in self.modules:
      if isinstance(module, nj.Module):
        param_list.append(module.values)
      elif isinstance(module, str):
        param_list.append({k: v for k, v in nj.context().items() if k.startswith(module)})
      elif isinstance(module, jax.Array):
        param_list.append(module)
      else:
        raise ValueError(module)
    metrics['param_rms'] = F.rms(param_list)
    metrics['param_count'] = jnp.array(list(counts.values()), f32).sum()
    metrics = {f'{self.name}/{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(i32(keep) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        f32(keep) * self.grad_scale.read() +
        f32(incr) * self.grad_scale.read() * 2 +
        f32(decr) * self.grad_scale.read() / 2, 1e-4, 1e5))
    return finite

  def _summarize_params(self, counts, depth):
    lines = []
    pfxs = []
    for key in counts:
      parts = key.split('/')
      pfxs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), depth))]
    subcounts = {
        prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
        for prefix in set(pfxs)}
    lines = [f'Optimizer {self.name} has {sum(counts.values()):,} params:']
    for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
      lines.append(f'{count:>14,} {prefix}')
    return '\n'.join(lines)


def clip_by_agc(clip=0.3, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = jax.tree.map(fn, params, updates) if clip else updates
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = jax.tree.map(
        lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = jax.tree.map(
        lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = jax.tree.map(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)


def build_optax_optimizer(
  lr: float = 4e-5,
  agc: float = 0.3,
  eps: float = 1e-20,
  beta1: float = 0.9,
  beta2: float = 0.999,
  nesterov: bool = False,
  wd: float = 0.0,
  wdregex: str = r'/weights$',
  schedule: str = 'const',
  warmup: int = 1000,
  anneal: int = 0,
):
  chain = []
  chain.append(clip_by_agc(agc))
  chain.append(scale_by_rms(beta2, eps))
  chain.append(scale_by_momentum(beta1, nesterov))
  if wd:
    assert not wdregex[0].isnumeric(), wdregex
    pattern = re.compile(wdregex)
    wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
    chain.append(optax.add_decayed_weights(wd, wdmask))
  assert anneal > 0 or schedule == 'const'
  if schedule == 'const':
    sched = optax.constant_schedule(lr)
  elif schedule == 'linear':
    sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
  elif schedule == 'cosine':
    sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
  else:
    raise NotImplementedError(schedule)
  if warmup:
    ramp = optax.linear_schedule(0.0, lr, warmup)
    sched = optax.join_schedules([ramp, sched], [warmup])
  chain.append(optax.scale_by_learning_rate(sched))
  opt: optax.GradientTransformationExtraArgs = optax.chain(*chain)
  return opt
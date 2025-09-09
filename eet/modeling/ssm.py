"""
File: ssm.py
Author: Viet Nguyen
Date: 2025-06-26

Description: Contain all types of encoder
"""


import jax
import jax.numpy as jnp
import numpy as np
from lib import nn
import functools

from lib import Space, print
from lib.nn import functional as F

class RSSM(nn.Module):

  deter: int = 64
  hidden: int = 64
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'silu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def __init__(self, **kw):
    assert self.deter % self.blocks == 0
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
      deter=Space(np.float32, self.deter),
      stoch=Space(np.float32, (self.stoch, self.classes))
    )

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def initial(self, bsize: int):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], nn.f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], nn.f32)))
    return carry

  def observe(self, carry, embed, reset, training, single=False):
    carry, embed = nn.cast((carry, embed))
    if single:
      carry, (entry, feat) = self._observe(
          carry, embed, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(embed)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nn.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (embed, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, embed, reset, training):
    deter, stoch = F.mask(
        (carry['deter'], carry['stoch']), ~reset)
    deter = self._core(deter, stoch)
    embed = embed.reshape((*deter.shape[:-1], -1))
    x = embed if self.absolute else jnp.concatenate([deter, embed], -1)
    logit = self._posterior(x)
    stoch = nn.cast(self._dist(logit).sample(seed=nn.seed()))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def _core(self, deter: jax.Array, stoch: jax.Array) -> jax.Array:
    # Given previous deterministic state and stochastic state,
    #   return the next deterministic
    # This serves as the specialized GRU for the SSM
    stoch = stoch.reshape((stoch.shape[0], -1))
    g = self.blocks
    flat2group = lambda x: jnp.reshape(x, x.shape[:-1] + (g, -1)) # ... (g h) -> ... g h
    group2flat = lambda x: jnp.reshape(x, x.shape[:-2] + (-1,)) # ... g h -> ... (g h)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.get_act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.get_act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x = jnp.concatenate([x0, x1], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    # GRU here
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, feat: jax.Array) -> jax.Array:
    # Given some feature vector, return the logit for the prior distribution
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _posterior(self, feat: jax.Array) -> jax.Array:
    # Given some feature vector, return the logit for the posterior distribution
    x = feat
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    return self._logit('obslogit', x)

  def _logit(self, name: str, x: jax.Array) -> jax.Array:
    # Utility function to create logits with name
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits: jax.Array) -> nn.distributions.Distribution:
    # Utility function to get the distribution given the logits
    out = nn.distributions.Agg(nn.distributions.OneHot(logits, self.unimix), 1, jnp.sum)
    return out

  def loss(self, carry, embed, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, embed, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(nn.sg(post)).kl(self._dist(prior))
    rep = self._dist(post).kl(self._dist(nn.sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics


class EBSSM(nn.Module):

  deter: int = 64
  hidden: int = 64
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'silu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 2
  dynlayers: int = 1
  alphalayers: int = 2
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def __init__(self, **kw):
    assert self.deter % self.blocks == 0
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
      deter=Space(np.float32, self.deter),
      stoch=Space(np.float32, (self.stoch, self.classes))
    )

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def initial(self, bsize: int):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], nn.f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], nn.f32)))
    return carry

  def observe(self, carry, embed, reset, training, single=False):
    carry, embed = nn.cast((carry, embed))
    if single:
      carry, (entry, feat, postlogit, priorlogit, alphalogit) = self._observe(
          carry, embed, reset, training)
      return carry, entry, feat, postlogit, priorlogit, alphalogit
    else:
      unroll = jax.tree.leaves(embed)[0].shape[1] if self.unroll else 1
      carry, (entries, feat, postlogit, priorlogit, alphalogit) = nn.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (embed, reset), unroll=unroll, axis=1)
      return carry, entries, feat, postlogit, priorlogit, alphalogit

  def _observe(self, carry, embed, reset, training):
    # alpha: how much to use posterior
    deter, stoch = F.mask(
        (carry['deter'], carry['stoch']), ~reset)
    deter = self._core(deter, stoch)
    embed = embed.reshape((*deter.shape[:-1], -1))
    x = embed if self.absolute else jnp.concatenate([deter, embed], -1)
    postlogit = self._posterior(x)
    priorlogit = self._prior(deter)
    poststoch = nn.cast(self._dist(postlogit).sample(seed=nn.seed()))
    priorstoch = nn.cast(self._dist(priorlogit).sample(seed=nn.seed()))
    assert poststoch.shape == priorstoch.shape, \
        f"poststoch: {poststoch.shape}, priorstoch: {priorstoch.shape}"
    alphalogit = self._alpha(embed, training)
    # Well this is weird and buggy
    alpha = jax.nn.sigmoid(alphalogit) # (B,)
    _alpha = F.expand_like(alpha, poststoch) # (B, stoch, classes)
    # _alpha = nn.sg(_alpha) # might want to try sg here
    stoch = _alpha * poststoch + (1 - _alpha) * priorstoch
    # stoch = poststoch
    carry = dict(deter=deter, stoch=poststoch)
    feat = dict(deter=deter, stoch=poststoch, combine=stoch)
    entry = dict(deter=deter, stoch=poststoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch))
    return carry, (entry, feat, postlogit, priorlogit, alphalogit)

  def _core(self, deter: jax.Array, stoch: jax.Array) -> jax.Array:
    # Given previous deterministic state and stochastic state,
    #   return the next deterministic
    # This serves as the specialized GRU for the SSM
    stoch = stoch.reshape((stoch.shape[0], -1))
    g = self.blocks
    flat2group = lambda x: jnp.reshape(x, x.shape[:-1] + (g, -1)) # ... (g h) -> ... g h
    group2flat = lambda x: jnp.reshape(x, x.shape[:-2] + (-1,)) # ... g h -> ... (g h)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.get_act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.get_act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x = jnp.concatenate([x0, x1], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    # GRU here
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, feat: jax.Array) -> jax.Array:
    # Given some feature vector, return the logit for the prior distribution
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _posterior(self, feat: jax.Array) -> jax.Array:
    # Given some feature vector, return the logit for the posterior distribution
    x = feat
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    return self._logit('obslogit', x)

  def _logit(self, name: str, x: jax.Array) -> jax.Array:
    # Utility function to create logits with name
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits: jax.Array) -> nn.distributions.Distribution:
    # Utility function to get the distribution given the logits
    out = nn.distributions.Agg(nn.distributions.OneHot(logits, self.unimix), 1, jnp.sum)
    return out

  def _alpha(self, embed: jax.Array, training: bool) -> jax.Array:
    # export a logit for alpha to be sigmoided
    # embed: (..., hidden)
    x = embed
    for i in range(self.alphalayers):
      x = self.sub(f'alpha{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.get_act(self.act)(self.sub(f'alpha{i}norm', nn.Norm, self.norm)(x))
    alpha = self.sub('alpha', nn.Linear, 1, **self.kw)(x)
    alpha = jnp.squeeze(alpha, -1)
    return alpha # (...)

  def loss(self, carry, embed, target_alpha, reset, training, eps=0.0001):
    # target_alpha: (B, T)
    # alpha control how much to weight the posterior
    # rmb, alpha should be sg
    metrics = {}
    carry, entries, feat, postlogit, priorlogit, alphalogit = self.observe(carry, embed, reset, training)
    # prior = self._prior(feat['deter'])
    # post = feat['logit']
    # dyn = self._dist(nn.sg(postlogit)).kl(self._dist(priorlogit))
    # rep = self._dist(postlogit).kl(self._dist(nn.sg(priorlogit)))
    # _alpha = jax.nn.sigmoid(alphalogit)  # (B, T)
    # _alpha = nn.sg(_alpha) # (B, T) # might want to try sg here
    # if self.free_nats:
    #   dyn = jnp.maximum(dyn, self.free_nats)
    #   rep = jnp.maximum(rep, self.free_nats)
    # Then, compute the loss for alpha
    alpha_loss = F.huber_loss(jax.nn.sigmoid(alphalogit), target_alpha)
    # alpha_loss = (jax.nn.sigmoid(alphalogit) - target_alpha)**2
    # Add regularization to encourage alpha to be close to 1.0
    # alpha_reg = (jax.nn.sigmoid(alphalogit) - 1.0)**2
    alpha_loss = alpha_loss # + alpha_reg  # Strong regularization towards a point
    # losses = {'dyn': dyn, 'rep': rep, 'alpha': alpha_loss}
    losses = {'alpha': alpha_loss}
    metrics['dyn_ent'] = self._dist(priorlogit).entropy().mean()
    metrics['rep_ent'] = self._dist(postlogit).entropy().mean()
    return carry, entries, losses, feat, metrics

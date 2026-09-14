import numpy as np
import torch
from torch import distributions as torchd
from torch import nn

from . import distributions as dists
from .base import BlockLinear, LambdaLayer

def weight_init_(m, fan_type="in"):
  # RMSNorm: initialize scale to 1.
  if isinstance(m, nn.RMSNorm):
    with torch.no_grad():
      m.weight.fill_(1.0)
    return

  weight = getattr(m, "weight", None)
  if weight is None:
    return

  if weight.numel() == 0:
    return

  # This is a torch private API, but widely used and stable.
  in_num, out_num = torch.nn.init._calculate_fan_in_and_fan_out(weight)

  with torch.no_grad():
    fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]
    std = 1.1368 * np.sqrt(1 / fan)
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    # set bias always 0
    bias = getattr(m, "bias", None)
    if bias is not None:
      bias.fill_(0.0)


def rpad(x, pad):
  for _ in range(pad):
    x = x.unsqueeze(-1)
  return x


class AISSM(nn.Module):
  def __init__(self, config, embed_size):
    super().__init__()
    self._stoch = int(config.stoch)
    self._deter = int(config.deter)
    self._hidden = int(config.hidden)
    self._discrete = int(config.discrete)
    act = getattr(torch.nn, config.act)
    self._unimix_ratio = float(config.unimix_ratio)
    self._device = torch.device(config.device)
    self._obs_layers = int(config.obs_layers)
    self._img_layers = int(config.img_layers)
    self._dyn_layers = int(config.dyn_layers)
    self._alphalayers = int(config.alphalayers)
    self._blocks = int(config.blocks)
    self._free = float(config.free_nats)
    self.flat_stoch = self._stoch * self._discrete
    self.feat_size = self.flat_stoch + self._deter
    self.out_dim = self.feat_size

    # Dynamic model
    assert self._deter % 2 == 0, "Deterministic state size must be divisible by 2 for bidirectional GRU."
    self._dyn_final = nn.GRU(input_size=embed_size,
      hidden_size=self._deter // 2, batch_first=True, bidirectional=True)

    # Posterior network
    self._obs_net = nn.Sequential()
    inp_dim = embed_size
    for i in range(self._obs_layers):
      self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
      self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
      self._obs_net.add_module(f"obs_net_a_{i}", act())
      inp_dim = self._hidden
    self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
    self._obs_net.add_module(
      "obs_net_lambda",
      LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
    )

    # Prior network
    self._img_net = nn.Sequential()
    inp_dim = self._deter
    for i in range(self._img_layers):
      self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
      self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
      self._img_net.add_module(f"img_net_a_{i}", act())
      inp_dim = self._hidden
    self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
    self._img_net.add_module(
      "img_net_lambda",
      LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
    )

    self._alpha_net = nn.Sequential()
    input_dim = embed_size
    for i in range(self._alphalayers):
      self._alpha_net.add_module(f"alpha_net_{i}", nn.Linear(input_dim, self._hidden, bias=True))
      self._alpha_net.add_module(f"alpha_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
      self._alpha_net.add_module(f"alpha_net_a_{i}", act())
      input_dim = self._hidden
    self._alpha_net.add_module("alpha_net_out", nn.Linear(input_dim, 1))

    # Weight init
    self.apply(weight_init_)

  def initial(self, batch_size):
    """Return an initial latent state."""
    # (B, D), (B, S, K)
    deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
    stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
    return stoch, deter

  def observe(self, embed, initial):
    """Posterior rollout using observations."""
    # (B, T, E), ((B, S, K), (B, D)) (B, T)
    L = embed.shape[1]
    post_stoch, _ = initial
    deters, _ = self._dyn_final(embed) # (B, T, D)
    post_stochs, post_logits, prior_stochs, prior_logits = [], [], [], []
    for i in range(L):
      # (B, S, K), (B, D), (B, S, K)
      post_stoch, post_logit, prior_stoch, prior_logit = self.obs_step(deters[:, i], embed[:, i])
      post_stochs.append(post_stoch)
      post_logits.append(post_logit)
      prior_stochs.append(prior_stoch)
      prior_logits.append(prior_logit)
    # (B, T, S, K), (B, T, D), (B, T, S, K)
    post_stochs = torch.stack(post_stochs, dim=1)
    post_logits = torch.stack(post_logits, dim=1)
    prior_stochs = torch.stack(prior_stochs, dim=1)
    prior_logits = torch.stack(prior_logits, dim=1)
    return deters, post_stochs, post_logits, prior_stochs, prior_logits

  def obs_step(self, deter, embed):
    """Single posterior step."""
    # (B, S, K), (B, D), (B, E), (B,)
    # Deterministic transition then posterior logits conditioned on embed.
    # (B, D)
    # (B, S, K)
    post_logit = self._obs_net(embed) # (B, S, K)

    # Sample discrete stochastic state via straight-through Gumbel-Softmax.
    # (B, S, K)
    post_stoch = self.get_dist(post_logit).rsample()
    prior_stoch, prior_logit = self.prior(deter)
    return post_stoch, post_logit, prior_stoch, prior_logit

  def prior(self, deter):
    """Compute prior distribution parameters and sample stoch."""

    # (B, S, K)
    logit = self._img_net(deter) # (B, S, K)
    stoch = self.get_dist(logit).rsample()
    return stoch, logit

  def alpha(self, embed: torch.Tensor) -> torch.Tensor:
    # export a logit for alpha to be sigmoided
    # embed: (..., hidden)
    x = embed
    for layer in self._alpha_net:
      x = layer(x)
    alpha = x.squeeze(-1)
    return alpha # (...)

  def get_feat(self, stoch, deter):
    """Flatten stoch and concatenate with deter."""
    # (B, S, K), (B, D)
    # (B, S*K)
    stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
    # (B, S*K + D)
    return torch.cat([stoch, deter], -1)

  def get_dist(self, logit):
    return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

  def loss(self, post_logit, prior_logit):
    # (B, T, S, K), (B, T, S, K)
    kld = dists.kl
    # (B, T)
    rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
    dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
    rep_loss = torch.clip(rep_loss, min=self._free)
    dyn_loss = torch.clip(dyn_loss, min=self._free)
    # (B, T)
    return dyn_loss, rep_loss

  def _dynamic(self, stoch, deter):
    """Deterministic state transition (block-GRU style)."""
    # (B, S, K), (B, D)
    B = stoch.shape[0]

    # Flatten stochastic state.
    # (B, S*K)
    stoch = stoch.reshape(B, -1)
    # (B, U)
    x0 = self._dyn_in0(deter)
    x1 = self._dyn_in1(stoch)

    # Concatenate projected inputs and broadcast over blocks.
    # (B, 2*U)
    x = torch.cat([x0, x1], -1)
    # (B, G, 2*U)
    x = x.unsqueeze(-2).expand(-1, self._blocks, -1)

    # Combine per-block deterministic state with per-block inputs.
    # (B, G, D/G + 2*U) -> (B, D + 2*U*G)
    x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

    # (B, D)
    x = self._dyn_hid(x)
    # (B, 3*D)
    x = self._dyn_gru(x)

    # Split GRU-style gates block-wise.
    # (B, G, 3*D/G)
    gates = torch.chunk(self.flat2group(x), 3, dim=-1)

    # (B, D)
    reset, cand, update = (self.group2flat(x) for x in gates)
    reset = torch.sigmoid(reset)
    cand = torch.tanh(reset * cand)
    update = torch.sigmoid(update - 1)
    # (B, D)
    return update * cand + (1 - update) * deter
"""
TDTracker: Temporal-Dual Tracker for Event-Based Eye Tracking

Implementation of TDTracker model for gaze estimation from event camera data.
This model combines 3D CNNs for spatiotemporal feature extraction, Bidirectional GRU
for temporal modeling, and Mamba (State Space Model) for long-range dependencies.

Architecture Overview:
- Input: Event voxel grid (Batch, 100, 2, 60, 80) - 100 temporal bins, 2 polarities, 60×80 spatial
- Encoder: 3-stage 3D CNN with spatial and temporal convolutions
- Temporal Modeling: Bidirectional GRU (256 hidden units)
- State Space Model: Mamba block for capturing long-range temporal dependencies
- Output: Dual heads for x and y coordinates (SimDR-style: 80 bins for x, 60 bins for y)

Performance (3ET+ 2025 Dataset):
- P3: 87.78% | P5: 96.08% | P10: 98.11% | Mean Error: ~16px (640×480 resolution)
- Parameters: 3.248M | FLOPs: 318M

Reference:
- Paper: "TDTracker: Temporal-Dual Tracker for Event-Based Eye Tracking"
- Dataset: 3ET+ 2025 Challenge (CVPR 2025)

Author: Mobina
Date: October 2025
"""

import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat


class Model(nn.Module):
    """
    TDTracker Main Model Class
    
    A deep learning model for event-based gaze estimation using voxel grid representation.
    Processes event data through 3D convolutions, bidirectional GRU, and Mamba blocks.
    
    Args:
        args: Optional configuration arguments (not currently used, kept for compatibility)
    
    Input Shape:
        x: (Batch, SeqLen=100, Channels=2, Height=60, Width=80)
           - SeqLen: Number of temporal bins
           - Channels: Event polarity (positive/negative)
           - Height, Width: Spatial resolution
    
    Output Shape:
        predict_w: (Batch, SeqLen, 80) - x-coordinate distribution
        predict_h: (Batch, SeqLen, 60) - y-coordinate distribution
    """
    
    def __init__(self, args=None):
        super().__init__() 
        self.args = args

        # ============ Stage 1: 3D CNN Encoder ============
        # First convolution: Extract spatial features (7x7 kernel)
        self.conv1_spatial  = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1,7,7), stride=(1,1,1), padding=(0,3,3)),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU()
        )
        # First temporal convolution: Capture short-term temporal dynamics
        self.conv1_temporal = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=(1,1,1), bias=False, dilation=(1,1,1)),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.AvgPool3d((1,3,3))  # Spatial downsampling: 60x80 -> 20x26
        )
          
        self.conv2_spatial = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
                                            nn.BatchNorm3d(num_features=64),
                                            nn.ReLU())
        self.conv2_temporal = nn.Sequential(nn.Conv3d(64, 64, kernel_size=(3,3,3),padding=(1,1,1),bias=False),
                                            nn.BatchNorm3d(num_features=64),
                                            nn.ReLU(),
                                            nn.AvgPool3d((1,3,3)))
        
        self.conv3_spatial = nn.Sequential(nn.Conv3d(64, 128, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
                                            nn.BatchNorm3d(num_features=128),
                                            nn.ReLU())
        self.conv3_temporal = nn.Sequential(nn.Conv3d(128, 128, kernel_size=(3,3,3),padding=(1,1,1),bias=False,dilation=(1,1,1)),
                                            nn.BatchNorm3d(num_features=128),
                                            nn.ReLU(),
                                            nn.Dropout())

        self.fft_sptaial_weight_3 = nn.Parameter(torch.cat((torch.ones(100//2+1,2048,1, dtype=torch.float32),torch.zeros(100//2+1,2048,1, dtype=torch.float32)),dim=-1))
        self.fft_sptaial_3= fftlayer_temporal(self.fft_sptaial_weight_3)

        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.AdaptiveAvgPool3d((128, 4, 4))
        self.spatialdropout = SpatialDropout(0.5)
        self.layernorm = nn.LayerNorm(2048)
        self.gru = nn.GRU(input_size=2048, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.mamba = ResidualBlock(MambaConfig(d_model=2 * 128))
        self.fc_1 = nn.Sequential(nn.Linear(128 * 2, 128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, 80),
                                )
        self.fc_2 = nn.Sequential(nn.Linear(128 * 2, 128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, 60),
                                )
    
    def forward(self, x,y=None):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1_spatial(x)
        x = self.conv1_temporal(x)
        x = self.conv2_spatial(x)
        x = self.conv2_temporal(x)
        x = self.conv3_spatial(x)
        x = self.conv3_temporal(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.pool(x)
        x = self.spatialdropout(x)
        x = x.reshape(batch_size,seq_len, -1)
        # x = self.fft_sptaial_3(x)
        # x = F.relu(x)
        x = self.layernorm(x)
        x, _ = self.gru(x)
        x = self.mamba(x)
        predict_w = self.fc_1(x)
        predict_h = self.fc_2(x)
        return predict_w, predict_h

class fftlayer_temporal(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.complex_weight = weight

    def forward(self, x):
        x = torch.fft.rfft(x, dim=(1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x, dim=(1), norm='ortho')
        return x

class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1]) 
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = False # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)
        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)
        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output

    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()
        # TODO remove .float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)
        
        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []
        
        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

# Placeholder pscan function - would need actual implementation or use sequential scan
def pscan(deltaA, BX):
    # For now, use sequential implementation
    B, L, ED, N = deltaA.shape
    h = torch.zeros(B, ED, N, device=deltaA.device)
    hs = []
    
    for t in range(L):
        h = deltaA[:, t] * h + BX[:, t]
        hs.append(h)
        
    return torch.stack(hs, dim=1)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)
        output = self.mixer(self.norm(x)) + x
        return output

if __name__ == '__main__':
    model = Model(args=None)
    print("TDTracker model created successfully")

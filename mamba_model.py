import math
import logging

import numpy as np 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from einops import rearrange, repeat

from selective_scan import selective_scan_ref

logger = logging.getLogger(__name__)



class mambaCofig:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class ssm(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.layer_idx = layer_idx
        
        dt_min = config.dt_min
        dt_max = config.dt_max
        dt_init = config.dt_init
        dt_scale = config.dt_scale
        dt_init_floor = config.dt_init_floor
        conv_bias = config.conv_bias
        in_out_proj_bias = config.in_out_proj_bias
        device = config.device
        dtype = config.dtype

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=in_out_proj_bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=in_out_proj_bias, **factory_kwargs)
        
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        seqlen = hidden_states.shape[1]

        conv_state, ssm_state = None, None

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=1)
            # Compute short convolution
        if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
        
        x = self.act(self.conv1d(x)[..., :seqlen])


        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_ref(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
        if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class mambaBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.ssm = ssm(config, layer_idx)
        self.norm_f = nn.LayerNorm(config.d_model, **factory_kwargs)

    def forward(self, hidden_states, residual=None, inference_params=None):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        hidden_states = self.ssm(hidden_states, inference_params=inference_params)
        return hidden_states, residual

class MambaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

        self.ssm_layers = nn.ModuleList(
            [
                mambaBlock(config, layer_idx=i)
                for i in range(config.n_layer)
            ]
        )

        self.norm_f = nn.LayerNorm(config.d_model, **factory_kwargs)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, **factory_kwargs)
        self.apply(self._init_weights)
        #self.tie_weights()
        logger.info("Number of parameters: %e",sum(p.numel() for p in self.parameters()))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def tie_weights(self):
        self.lm_head.weight = self.embedding.weight
    
    def mamba_optimizer(self, train_config):

        decay = set()
        whitelist_weight_modules = (torch.nn.Linear)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" %(mn, pn) if mn else pn
                if pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
 
        param_dict = {pn: p  for pn, p in self.named_parameters()}
        
        optim_groups = [
            {"params":[param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params":[param_dict[pn] for pn in param_dict.keys() if pn not in sorted(list(decay))], "weight_decay": 0.0}
            ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.AdamwBetas)
        return optimizer

    
    def forward(self, input_ids, targets=None, inference_params=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.ssm_layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
       
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        logits = self.lm_head(hidden_states)

        loss=None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate(self, idx, num_tokens):
        tokens_len = idx.shape[1]
        for _ in range(num_tokens):
            block_idx = idx[: , -self.config.block_size:]
            logits, _ = self.forward(block_idx)
            logits = logits.reshape(1, tokens_len, -1)[:,-1,:]
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if tokens_len < self.config.block_size:
                tokens_len+=1
        return idx


if __name__ == "__main__":
    config = mambaCofig(
        vocab_size=10,
        d_model = 2,
        d_state = 5,
        d_conv = 5,
        expand = 2,
        n_layer=3,
        device="cpu",
        dtype = torch.float32,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        in_out_proj_bias=False,
        block_size=5,
    )
    model = MambaLM(config)
    u = torch.randint(low=0, high=10, size=(5, 3))
    y, _ = model(u)
    assert y.shape == (5,3,10)

    idx = torch.zeros((1,1), dtype=torch.long).to("cpu")
    generated_tokens = model.generate(idx ,5)
    print(f"generated_token idx: {generated_tokens}")
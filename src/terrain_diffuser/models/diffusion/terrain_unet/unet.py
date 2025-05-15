"""
Top-level ColorMap UNet that plugs residual/attention blocks together.
"""
from __future__ import annotations
from typing import Dict, List
import torch
from torch import nn
from terrain_diffuser.models.diffusion.terrain_unet.sinusoidal_embeddings import SinusoidalEmbedding
from terrain_diffuser.models.diffusion.terrain_unet.resblock import UnetResBlock
from terrain_diffuser.models.diffusion.terrain_unet.utils import Conv2d

__all__ = ["ColorMapUnet"]


class ColorMapUnet(nn.Module):
    def __init__(
        self,
        *,
        img_resolution: int = 32,
        in_channels: int = 3,
        out_channels: int = 3,
        text_dim: int = 512,
        ngroups: int = 16,
        model_channels: int = 128,
        channel_mult: List[int] = (2, 2, 2),
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = (16,),
        feature_channels: Dict[str, int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.channel_mult = list(channel_mult)

        # --- time embedding -----------------------
        time_embed_dim = model_channels * channel_mult_emb
        self.time_embedding = SinusoidalEmbedding(model_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- input conv ---------------------------
        self.input_conv = Conv2d(in_channels, model_channels * channel_mult[0], 3)

        # --- encoder ------------------------------
        self.encoder = nn.ModuleList()
        ch_stack: List[int] = [model_channels * channel_mult[0]]

        for lvl, mult in enumerate(channel_mult):
            res = img_resolution >> lvl
            cin = model_channels * mult

            # optional downsample at start of each new resolution (except first)
            if lvl > 0:
                self.encoder.append(
                    UnetResBlock(
                        ch_stack[-1],
                        cin,
                        ngroups,
                        t_emb_dim=time_embed_dim,
                        c_emb_dim=text_dim,
                        dropout=dropout,
                        down=True,
                    )
                )
                ch_stack.append(cin)

            for _ in range(num_blocks):
                self.encoder.append(
                    UnetResBlock(
                        ch_stack[-1],
                        cin,
                        ngroups,
                        t_emb_dim=time_embed_dim,
                        c_emb_dim=text_dim,
                        dropout=dropout,
                        feature_channels=(feature_channels or {}).get(str(res)),
                        attn=(res in attn_resolutions),
                    )
                )
                ch_stack.append(cin)

        # --- middle ----------------------------------
        self.middle = nn.ModuleList(
            [
                UnetResBlock(
                    ch_stack[-1],
                    ch_stack[-1],
                    ngroups,
                    t_emb_dim=time_embed_dim,
                    c_emb_dim=text_dim,
                    dropout=dropout,
                    feature_channels=(feature_channels or {}).get(str(img_resolution >> (len(channel_mult) - 1))),
                    attn=((img_resolution >> (len(channel_mult) - 1)) in attn_resolutions),
                ),
                UnetResBlock(
                    ch_stack[-1],
                    ch_stack[-1],
                    ngroups,
                    t_emb_dim=time_embed_dim,
                    c_emb_dim=text_dim,
                    dropout=dropout,
                ),
            ]
        )

        # --- decoder ---------------------------------
        self.decoder = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> lvl
            for _ in range(num_blocks + 1):
                self.decoder.append(
                    UnetResBlock(
                        ch_stack[-1] + ch_stack.pop(),  # skip connection
                        model_channels * mult,
                        ngroups,
                        t_emb_dim=time_embed_dim,
                        c_emb_dim=text_dim,
                        dropout=dropout,
                        feature_channels=(feature_channels or {}).get(str(res)),
                        attn=(res in attn_resolutions),
                    )
                )
            if lvl > 0:
                self.decoder.append(
                    UnetResBlock(
                        ch_stack[-1],
                        model_channels * channel_mult[lvl - 1],
                        ngroups,
                        t_emb_dim=time_embed_dim,
                        c_emb_dim=text_dim,
                        dropout=dropout,
                        up=True,
                    )
                )

        # --- output -----------------------------------
        self.output = nn.Sequential(
            nn.GroupNorm(ngroups, model_channels * channel_mult[0]),
            nn.SiLU(),
            Conv2d(model_channels * channel_mult[0], out_channels, 3),
        )

    # --------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        content_features: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        content_features = content_features or {}
        t = self.time_mlp(self.time_embedding(timesteps))

        h = self.input_conv(x)
        hs = [h]

        # --- encoder pass --------------------------
        for blk in self.encoder:
            res_str = str(h.shape[-1])
            h = blk(h, t, text_emb, content_features.get(res_str))
            hs.append(h)

        # --- middle --------------------------------
        for blk in self.middle:
            res_str = str(h.shape[-1])
            h = blk(h, t, text_emb, content_features.get(res_str))

        # --- decoder pass --------------------------
        for blk in self.decoder:
            if isinstance(blk, UnetResBlock) and not (blk.up or blk.down):
                h = torch.cat([h, hs.pop()], dim=1)
            res_str = str(h.shape[-1])
            h = blk(h, t, text_emb, content_features.get(res_str))

        return self.output(h)

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import dataclasses

from layers.sinusoidal_emb import SinusoidalEmbedding
from layers.unet_block import UnetBlock

class TerrainDiffuser(nn.Module):
    out_channels: int = 3
    img_resolution: int = 32
    num_groups: int = 16
    model_channels: int = 128
    channel_mult: Sequence[int] = (2, 2, 2)
    time_emb_mult: int = 4
    num_blocks: Sequence[int] = (4, 4, 4)
    cond_resolutions: dict = dataclasses.field(default_factory=lambda: {"16": 256})
    p_dropout: float = 0.1
    epsilon: float = 1e-5
    reduction: int = 16
    use_bias: bool = False
    num_heads: int = 1
    skip_scaling: float = 1/jnp.sqrt(2)
    
    def _extract_feat_maps(self, feat_maps, resolution):
        """
        Extracts the feature tensor from the input feat_maps NCHW to NHWC unpadded tensor corresponding to resolution
        """
        feature_channels = self.cond_resolutions.get(str(resolution))
        if feature_channels is None:
            return None
        start_index = 0
        for key, value in sorted(self.cond_resolutions.items(), key=lambda kv: int(kv[0]),  reverse=True):
            if int(key) > resolution:
                start_index += value
            else:
                break
            
        res_feat_maps = feat_maps[:, :resolution, :resolution, start_index:start_index+feature_channels]
        return res_feat_maps
    
    @nn.compact
    def __call__(self, x, timesteps, text_emb, feat_maps, train:bool=True):
        """
        NOTE: feat_maps will be one big jnp array with shape (NHWC) where The resolutions will be padded to img_resolution
        and stacked across the channel dimension (dim=3). The feat_maps must be concated in descending resolution
        """
        
        time_emb = SinusoidalEmbedding(d_model=self.model_channels, max_time=10000)(timesteps)
        time_emb = nn.Sequential([
            nn.Dense(features=self.model_channels * self.time_emb_mult),
            nn.silu,
            nn.Dense(features=self.model_channels * self.time_emb_mult),
        ])(time_emb)
        
        skip_connections = []
        
        x = nn.Conv(features=self.model_channels * self.channel_mult[0], kernel_size=3)(x)
        skip_connections.append(x)
        
        # Down blocks for the Unet
        for level, mult in enumerate(self.channel_mult):
            resolution =  self.img_resolution >> level
            # Downsample when level is not 0
            if level > 0:
                x = UnetBlock(
                    model_channels=self.model_channels * mult,
                    p_dropout=self.p_dropout,
                    up=False,
                    down=True,
                    attn=False,
                    feature_channels=None,
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    use_bias=self.use_bias,
                    reduction=self.reduction,
                    num_heads=self.num_heads
                )(x, time_emb, text_emb, feat_maps, train)
                skip_connections.append(x)
                
            # Residual blocks with optional attention
            for _ in range(self.num_blocks[level]):
                x = UnetBlock(
                    model_channels=self.model_channels * mult,
                    p_dropout=self.p_dropout,
                    up=False,
                    down=False,
                    attn=(resolution in self.cond_resolutions),
                    feature_channels=(self.cond_resolutions.get(str(resolution))),
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    use_bias=self.use_bias,
                    reduction=self.reduction,
                    num_heads=self.num_heads
                )(x, time_emb, text_emb, self._extract_feat_maps(feat_maps, resolution), train)
                skip_connections.append(x)
                
        # Middle blocks for the Unet
        x = UnetBlock(
            model_channels=self.model_channels * mult,
            p_dropout=self.p_dropout,
            up=False,
            down=False,
            attn=(resolution in self.cond_resolutions),
            feature_channels=(self.cond_resolutions.get(str(resolution))),
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            reduction=self.reduction,
            num_heads=self.num_heads
        )(x, time_emb, text_emb, self._extract_feat_maps(feat_maps, resolution), train)
        x = UnetBlock(
            model_channels=self.model_channels * mult,
            p_dropout=self.p_dropout,
            up=False,
            down=False,
            attn=False,
            feature_channels=None,
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            reduction=self.reduction,
            num_heads=self.num_heads
        )(x, time_emb, text_emb, feat_maps, train)
        
        # Up blocks for the Unet
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            resolution = self.img_resolution >> level
            
            # Residual blocks with optional attention
            for _ in range(self.num_blocks[level] + 1):
                x = UnetBlock(
                    model_channels=self.model_channels * mult,
                    p_dropout=self.p_dropout,
                    up=False,
                    down=False,
                    attn=(resolution in self.cond_resolutions),
                    feature_channels=(self.cond_resolutions.get(str(resolution))),
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    use_bias=self.use_bias,
                    reduction=self.reduction,
                    num_heads=self.num_heads
                )(jnp.concat((x, skip_connections.pop() * self.skip_scaling), axis=-1), time_emb, text_emb, self._extract_feat_maps(feat_maps, resolution), train)
            
            # Upsample when level is not 0
            if level > 0:
                x = UnetBlock(
                    model_channels=self.model_channels * mult,
                    p_dropout=self.p_dropout,
                    up=True,
                    down=False,
                    attn=(resolution in self.cond_resolutions),
                    feature_channels=(self.cond_resolutions.get(str(resolution))),
                    num_groups=self.num_groups,
                    epsilon=self.epsilon,
                    use_bias=self.use_bias,
                    reduction=self.reduction,
                    num_heads=self.num_heads
                )(x, time_emb, text_emb, self._extract_feat_maps(feat_maps, resolution), train)
                
        x = nn.Sequential([
            nn.GroupNorm(num_groups=self.num_groups),
            nn.silu,
            nn.Conv(features=self.out_channels, kernel_size=3, use_bias=True)
        ])(x)
        
        return x
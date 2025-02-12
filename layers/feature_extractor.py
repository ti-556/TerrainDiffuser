import dataclasses

import jax 
import jax.numpy as jnp
import flax.linen as nn
import flaxmodels as fm 

class VGGFeatureExtractor(nn.Module):
    resolutions: dict = dataclasses.field(default_factory=lambda: {"16": 256})
    img_resolution: int = 32
    
    def setup(self):
        """
        Returns a dict of key conv name, value feature map output.
        relu1_1: 64 channels, 1x res_mult
        relu1_2: 64 channels, 1x res_mult
        relu2_1: 128 channels, 1/2x res_mult
        relu2_2: 128 channels, 1/2x res_mult
        relu3_1: 256 channels, 1/4x res_mult
        relu3_2: 256 channels, 1/4x res_mult
        relu3_3: 256 channels, 1/4x res_mult
        relu4_1: 512 channels, 1/8x res_mult
        relu4_2: 512 channels, 1/8x res_mult
        relu4_3: 512 channels, 1/8x res_mult
        relu5_1: 512 channels, 1/16x res_mult
        relu5_2: 512 channels, 1/16x res_mult
        relu5_3: 512 channels, 1/16x res_mult
        """
        self.vgg = fm.VGG16(output="activations", include_head=False, pretrained="imagenet")
        
    def __call__(self, x):
        """
        Returns a padded jnp array (filled from top left) in descending order of resolutions based on the resolutions dict initialized.
        """
        max_resolutions = max([int(k) for k in self.resolutions])
        channels = sum(self.resolutions.values())
        feat_maps = jnp.zeros(shape=(x.shape[0], max_resolutions, max_resolutions, channels))
        
        activations = self.vgg(x)
        res_acts = {}
        for name, activation in activations.items():
            if not name.startswith("relu"):
                continue

            resolution = str(self.img_resolution >> (int(name[4]) - 1))
            if self.resolutions.get(resolution):
                res_acts.setdefault(resolution, []).append(activation)
                
        channel_offset = 0
        for resolution, activations in res_acts.items():
            target_channels = self.resolutions.get(resolution)
            feat_maps = feat_maps.at[:, :int(resolution), :int(resolution), channel_offset:channel_offset+target_channels].set(
                jnp.concat(activations, axis=-1)[:,:,:,:self.resolutions.get(resolution)]
            )
            channel_offset += target_channels
        return feat_maps
    
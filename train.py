import logging
import json
import time
import argparse

import numpy as np
import torch
import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training.train_state import TrainState
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

from layers.terrain_diffuser import TerrainDiffuser
from layers.feature_extractor import VGGFeatureExtractor
from dataset.map_dataset import MapDataset


with open("./logconfigs.json", 'r') as f:
    cfgs = json.load(f)
    
logging.config.dictConfig(cfgs)
logger = logging.getLogger(__name__)

def create_train_states(key, model, vgg, tx, cfg):
    # Dummy inputs
    x = jnp.zeros((cfg.batch_size, cfg.img_resolution, cfg.img_resolution, 3))
    xf = jnp.zeros((cfg.batch_size, cfg.img_resolution, cfg.img_resolution, 3))
    feat_maps = jnp.zeros((cfg.batch_size, cfg.img_resolution, cfg.img_resolution, sum(cfg.cond_resolutions.values())))
    timesteps = jnp.zeros((cfg.batch_size,))
    text_emb = jnp.zeros((cfg.batch_size, cfg.text_len, cfg.text_dim))
    
    key, model_params, model_dropout, vgg_params, vgg_dropout = jax.random.split(key, 5)
    logger.info("Initializing model...")
    model_params = model.init({"params": model_params, "dropout": model_dropout}, x, timesteps, text_emb, feat_maps)
    logger.info("Initializing VGG image encoder...")
    vgg_params = vgg.init({"params": vgg_params, "dropout": vgg_dropout}, xf)
    logger.info("Initializing TrainState...")
    model_state = TrainState.create(apply_fn=model.apply, params=model_params["params"], tx=tx)
    ema_params = jax.tree_util.tree_map(jnp.array, model_params["params"])

    return model_state, vgg_params, ema_params, key

@jax.jit
def train_one_step(key, vgg, model_state, vgg_params, ema_params, dem, sat, text_emb, cfg):
    def sd3_timesampling(num_samples, key):
        t = jax.random.normal(key, shape=(num_samples,))
        t = 1 / (1 + jnp.exp(-t))
        return jnp.clip(t, 0.001, 0.999).reshape(num_samples, 1, 1, 1)
    
    def loss_fn(params):
        key, dropout_key, time_key, noise_key, t_cfgkey, f_cfgkey = jax.random.split(key, 6)
        feat_maps = vgg.apply(vgg_params, dem, train=False)
        t = sd3_timesampling(sat.shape[0], time_key)
        noise = jax.random.normal(noise_key, sat.shape)
        z_t = (1 - t) * sat + t * noise
        vector_field = sat - noise # models the vector field -dx, or the noise to data distribution
        if jax.random.uniform(t_cfgkey, shape=(1,)) < cfg.text_dropout: # Text CFG dropout
            text_emb = jnp.zeros(text_emb.shape)
        if jax.random.uniform(f_cfgkey, shape=(1,)) < cfg.feat_dropout: # feature CFG dropout
            feat_maps = jnp.zeros(feat_maps.shape)        
        
        pred = model_state.apply_fn(
            {"params": model_state.params},
            z_t,
            t,
            text_emb,
            feat_maps,
            train=True,
            rngs={"dropout": dropout_key}
        )
        loss = jnp.mean((pred - vector_field)**2)
        return loss, key
      
    loss, grads = jax.value_and_grad(fun=loss_fn, has_aux=True)(model_state.params)
    model_state = model_state.apply_gradients(grads=grads)
    # Applying ema update
    ema_params = jax.tree_util.tree_map(lambda e, p: e * cfg.decay_rate + (1 - cfg.decay_rate) * p, ema_params, model_state.params)
    
    return model_state, ema_params, loss, key
    
def train_one_epoch(key, vgg, model_state, vgg_params, ema_params, dataloader, cfg):
    for dem, sat, text_emb in tqdm(dataloader):
        model_state, ema_params, loss, key = train_one_step(
            key=key,
            vgg=vgg,
            model_state=model_state,
            vgg_params=vgg_params,
            ema_params=ema_params,
            dem=dem,
            sat=sat,
            text_emb=text_emb,
            cfg=cfg,
        )
        if cfg.log_wandb:
            wandb.log({"loss": loss.item()})
            
    return model_state, ema_params, loss, key
            
@jax.jit
def sample_one_step(model, params, z_t, dt, t, text_emb, feat_map, text_scale, feat_scale):
    # CFG guided vector field is v_guided = v_uncond + w1 * (v_cond1 - v_uncond) + w2 * (v_cond2 - v_uncond)
    null_text = jnp.zeros(shape=text_emb.shape)
    null_feat = jnp.zeros(shape=feat_map.shape)
    
    uncond_pred = model.apply(params, z_t, t, null_text, null_feat, train=False)
    text_pred = model.apply(params, z_t, t, text_emb, null_feat, train=False)
    feat_pred = model.apply(params, z_t, t, null_text, feat_map, train=False)
    
    guided_pred = uncond_pred + text_scale * (text_pred - uncond_pred) + feat_scale * (feat_pred - uncond_pred)
    # Note the model is techincally -dx where it goes from noise to signal so add
    z_t += guided_pred * dt
    return z_t
    
def ddim_eval(key, model, params, text_emb, feat_map, text_scale, feat_scale, cfg):
    batch_size = feat_map.shape[0]
    key, noise_key = jax.random.split(key)
    z_t = jax.random.normal(noise_key, shape=(batch_size, cfg.img_resolution, cfg.img_resolution, 3))
    for t in jnp.linspace(1, 0, cfg.sample_steps):
        ts = jnp.full(shape=(batch_size,), fill_value=t)
        dt = 1 / cfg.sample_steps
        z_t = sample_one_step(model, params, z_t, ts, dt, text_emb, feat_map, text_scale, feat_scale)
        
    return key, jnp.clip(z_t, -1, 1)

def evaluate(key, step, model, ema_params, vgg, vgg_params, dataloader, cfg):
    images_dict = {}
    for text_scale in cfg.text_scales:
        for feat_scale in cfg.feat_scales:
            for dem, _, text_emb in dataloader:
                feat_map = vgg.apply(vgg_params, dem, train=False)
                key, samples = ddim_eval(
                    key=key,
                    model=model,
                    params=ema_params,
                    text_emb=text_emb,
                    feat_map=feat_map,
                    text_scale=text_scale,
                    feat_scale=feat_scale,
                    cfg=cfg,
                )
                # log images to wandb
                grid = make_grid(torch.tensor(np.array(samples)), nrow=8, normalize=True)
                image = wandb.Image(grid, caption=f"Step: {step}")
                images_dict[f"Text CFG: {text_scale}, Feat CFG: {feat_scale}"] = image
                
    wandb.log(images_dict)
                

def train(cfg):
    if cfg.log_wandb and not cfg.dry_run:
        wandb.init(project=cfg.project_name, name=cfg.run_name)
        wandb.config.update(cfg)
    
    model = TerrainDiffuser(
        out_channels=cfg.out_channels,
        img_resolution=cfg.img_resolution,
        num_groups=cfg.num_groups,
        model_channels=cfg.model_channels,
        channel_mult=cfg.channel_mult,
        time_emb_mult=cfg.time_emb_mult,
        num_blocks=cfg.num_blocks,
        cond_resolutions=cfg.cond_resolutions,
        p_dropout=cfg.p_dropout,
        epsilon=cfg.epsilon,
        reduction=cfg.reduction,
        use_bias=cfg.use_bias,
        num_heads=cfg.num_heads,
        skip_scaling=cfg.skip_scaling,
    )
    vgg = VGGFeatureExtractor(resolutions=cfg.cond_resolutions, img_resolution=cfg.img_resolution)
    tx = optax.adamw(learning_rate=cfg.lr)
    key = jax.random.PRNGKey(seed=cfg.seed)
    
    model_state, vgg_params, ema_params, key = create_train_states(key=key, model=model, vgg=vgg, tx=tx, cfg=cfg)
        
    if not cfg.dry_run:
        # Setup dataloaders
        dataset = MapDataset(cfg.dem_dir, cfg.sat_dir, cfg.text_dir, cfg.img_resolution)
        train_dataset = torch.utils.data.Subset(dataset, indeices=range(cfg.test_size))
        test_dataset = torch.utils.data.Subset(dataset, indices=(cfg.test_size, len(dataset)))
        
        trainloader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        testloader = DataLoader(test_dataset, cfg.test_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        
        for epoch in range(cfg.epochs):
            start_time = time.time()
            model_state, ema_params, loss, key = train_one_epoch(key, vgg, model_state, vgg_params, ema_params, trainloader, cfg)
            logger.info(f"Epoch {epoch + 1} Finished. Current step: {model_state.step}. Time taken: {time.time()-start_time:.2f}s. Loss: {loss:.3f}")
            if (epoch + 1) % cfg.eval_every == 0:
                evaluate(
                    key=key,
                    step=model_state.step,
                    model=model,
                    ema_params=ema_params,
                    vgg=vgg,
                    vgg_params=vgg_params,
                    dataloader=testloader,
                    cfg=cfg
                )
    else:
        logger.debug("Dry run complete!")
    
    if cfg.log_wandb:
        wandb.finish()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training configs
    parser.add_argument("--log_wandb", type=lambda x: x.lower() == "true", choices=[True, False], default=False, help="Log to wandb")
    parser.add_argument("--project_name", type=str, required=False, help="Name of project to log in wandb")
    parser.add_argument("--run_name", type=str, required=False, help="Name of run to log in wandb")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train")
    parser.add_argument("--dry_run", type=lambda x: x.lower() == "true", choices=[True, False], default=False, help="Whether to dry run")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size of model training")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNG keys")
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--text_dropout", type=float, required=False, default=0.1, help="text embedding dropout rate for cfg")
    parser.add_argument("--feat_dropout", type=float, required=False, default=0, help="feature embedding dropout rate for cfg")
    parser.add_argument("--decay_rate", type=float, required=False, default=0.999, help="EMA decay rate during training")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate and save images to wandb every n epochs")
    
    # Data configs
    parser.add_argument("--dem_dir", type=str, required=True, help="Path to dem data directory")
    parser.add_argument("--sat_dir", type=str, required=True, help="Path to sat data directory")
    parser.add_argument("--text_dir", type=str, required=True, help="Path to text data directory")
    
    # Eval configs
    parser.add_argument("--text_scales", type=lambda s: [float(item) for item in s.split(',')], default="1,5,7", help="Comma-separated text cfg scales, e.g., '1,5,7'")
    parser.add_argument("--feat_scales", type=lambda s: [float(item) for item in s.split(',')], default="1,5,7", help="Comma-separated feat cfg scales, e.g., '1,5,7'")
    parser.add_argument("--sample_steps", type=int, default=100, help="Number of DDIM sample steps")
    parser.add_argument("--test_size", type=int, default=8, help="Number of DDIM sample steps")

    # model configs
    parser.add_argument("--out_channels", type=int, required=False, default=3, help="Number of output channels")
    parser.add_argument("--img_resolution", type=int, required=False, default=32, help="Input image resolution")
    parser.add_argument("--num_groups", type=int, required=False, default=16, help="Channels per group for GroupNorm")
    parser.add_argument("--model_channels", type=int, required=False, default=128, help="Number of base channels per Unet level")
    parser.add_argument("--channel_mult", type=lambda s: [int(item) for item in s.split(',')], default="2,2,2", help="Comma-separated channel mult per Unet level, e.g., '2,2,2'")
    parser.add_argument("--time_emb_mult", type=int, required=False, default=4, help="Time emb mult for intermediate layer injection")
    parser.add_argument("--num_blocks", type=lambda s: [int(item) for item in s.split(',')], default="4,4,4", help="Comma-separated number of Unet Blocks per level, e.g., '4,4,4'")
    parser.add_argument("--cond_resolutions", type=lambda s: json.loads(s), default='{"16": 256}', help="Number of feature channels per resolution")
    parser.add_argument("--p_dropout", type=float, required=False, default=0.1, help="Dropout probability")
    parser.add_argument("--epsilon", type=float, required=False, default=1e-5, help="Epsilon for numerical stability")
    parser.add_argument("--reduction", type=int, required=False, default=16, help="Squeeze and excite reduction factor")
    parser.add_argument("--use_bias", type=lambda x: x.lower() == "true", choices=[True, False], help="Use bias in QKV projections")
    parser.add_argument("--num_heads", type=int, required=False, default=1, help="Number of attention heads")
    parser.add_argument("--skip_scaling", type=int, required=False, default=1, help="Unet skip connection scaling")
    parser.add_argument("--text_dim", type=int, required=True, help="Text embedding hidden dimension")
    parser.add_argument("--text_len", type=int, required=False, default=77, help="The maximum sequence token length of text conditioning")
    
    cfg = parser.parse_args()
    
    train(cfg)
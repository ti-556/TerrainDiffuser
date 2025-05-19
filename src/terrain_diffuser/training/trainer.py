# terrain_diffuser/training/trainer.py
from __future__ import annotations

import random, copy, yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from contextlib import nullcontext
from typing import ContextManager, Callable

import torch, wandb
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

from terrain_diffuser.data.map_dataset               import MapDataset
from terrain_diffuser.models.image.vgg               import VGG16Encoder
from terrain_diffuser.models.text.t5                 import T5TextEncoder
from terrain_diffuser.models.diffusion.terrain_unet  import unet as unet_mod
from terrain_diffuser.training.weightings            import make_weighting
from terrain_diffuser.training.losses                import make_loss

# ──────────────────────────────────────────────────────────────── #
@dataclass
class TrainCfg:
    # ---- basics ----
    seed          : int   = 123
    device        : str   = "cuda"      # "cpu" to override
    precision     : str   = "fp16"      # fp32 | fp16 | bf16

    # ---- data paths ----
    dem_dir       : str   = "/home/chs/terraindiffuser/augmented/DEM"
    sat_dir       : str   = "/home/chs/terraindiffuser/augmented/SAT/SAT32"
    h5_path       : str   = "/home/chs/terraindiffuser/vectorlabels1_t5small.h5"
    val_size      : int   = 32          # first N samples → validation
    num_workers     : int = 4     
    prefetch_factor : int = 2            

    # ---- VGG feature extractor ----
    feature_channels : dict[int,int] = field(default_factory=lambda:{16:16})
    vgg_mode      : str   = "random"    # all | topk | bottomk | random
    vgg_seed      : int|None = None

    # ---- UNet ----
    unet          : dict  = field(default_factory=dict)  # filled from YAML

    # ---- text encoder ----
    text_model    : str   = "google/flan-t5-small"

    # ---- optimisation ----
    batch         : int   = 128
    epochs        : int   = 25
    lr            : float = 5e-4
    ema_decay     : float = 0.997
    cfg_drop_prob : float = 0.1

    # ---- algorithms ----
    weighting     : str   = "sd3"       # sd3 | cosine
    loss          : str   = "flow"      # flow | eps | xpred
    sampler       : str   = "sd3"       # sd3 | uniform

    # ---- logging / I/O ----
    wandb_proj    : str   = "TerrainGen"
    wandb_run     : str   = "run"
    out_dir       : str   = "saves"
    val_every     : int   = 1
    save_every    : int   = 10

    @classmethod
    def from_yaml(cls, path: str|Path):
        return cls(**yaml.safe_load(Path(path).read_text()))

# ──────────────────────────────────────────────────────────────── #
def _sample_timesteps(policy: str, batch: int, device):
    policy = policy.lower()
    if policy in ("sd3", "logistic"):
        z = torch.randn(batch, device=device)
        return (1 / (1 + torch.exp(-z))).clamp_(0.001, 0.999)
    if policy in ("uniform", "linear"):
        return torch.rand(batch, device=device).clamp_(0.001, 0.999)
    raise ValueError(f"Unknown sampler '{policy}'")

# ──────────────────────────────────────────────────────────────── #
class Trainer:
    # ============================================================== INIT
    def __init__(self, cfg: TrainCfg):
        self.cfg    = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        random.seed(cfg.seed); torch.manual_seed(cfg.seed)

        # ---------- precision / autocast ------------------------
        prec = cfg.precision.lower()
        if prec == "fp32":
            self._amp: Callable[[],ContextManager] = nullcontext
            self._amp_dtype = None
        elif prec == "bf16":
            self._amp = lambda: autocast(dtype=torch.bfloat16)
            self._amp_dtype = torch.bfloat16
        else:                              # default fp16
            self._amp = lambda: autocast(dtype=torch.float16)
            self._amp_dtype = torch.float16

        # ---------- encoders ------------------------------------
        self.txt_enc = T5TextEncoder(cfg.text_model).to(self.device)
        txt_dim      = self.txt_enc.model.config.hidden_size

        self.vgg_enc = VGG16Encoder(
            cfg.feature_channels,
            seed       = cfg.vgg_seed,
            pretrained = True,
            device     = self.device,
        )

        # ---------- UNet ----------------------------------------
        u = cfg.unet
        self.model = unet_mod.ColorMapUnet(
            img_resolution   = u.get("img_resolution",   32),
            in_channels      = u.get("in_channels",      3),
            out_channels     = u.get("out_channels",     3),
            text_dim         = txt_dim,
            ngroups          = u.get("ngroups",          16),
            model_channels   = u.get("model_channels",   128),
            channel_mult     = u.get("channel_mult",     (2,2,2)),
            channel_mult_emb = u.get("channel_mult_emb", 4),
            num_blocks       = u.get("num_blocks",       4),
            attn_resolutions = u.get("attn_resolutions",(16,)),
            feature_channels = {str(k):v for k,v in cfg.feature_channels.items()},
            dropout          = u.get("dropout",          0.10),
        ).to(self.device)

        # ---------- EMA -----------------------------------------
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)

        # ---------- dataset -------------------------------------
        tfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x*2-1),
        ])
        ds          = MapDataset(cfg.dem_dir, cfg.sat_dir, cfg.h5_path, tfm)
        va_ds       = torch.utils.data.Subset(ds, range(cfg.val_size))
        tr_ds       = torch.utils.data.Subset(ds, range(cfg.val_size, len(ds)))

        self.tr_loader = DataLoader(tr_ds, batch_size=cfg.batch, shuffle=True,
                                    num_workers=cfg.num_workers, pin_memory=True, prefetch_factor=cfg.prefetch_factor)
        self.va_loader = DataLoader(va_ds, batch_size=cfg.val_size)

        # ---------- optimisation -------------------------------
        scaler_enabled = (self._amp_dtype is torch.float16)  # GradScaler only for fp16
        self.scaler  = GradScaler(enabled=scaler_enabled)
        self.opt     = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.weight  = make_weighting(cfg.weighting)
        self.loss_fn = make_loss(cfg.loss)
        self.sample  = lambda b: _sample_timesteps(cfg.sampler, b, self.device)

        # ---------- logging -------------------------------------
        wandb.init(project=cfg.wandb_proj, name=cfg.wandb_run, config=asdict(cfg))

    # ============================================================== TRAIN LOOP
    def _training_step(self, dem, sat, text):
        dem, sat, text = dem.to(self.device), sat.to(self.device), text.to(self.device)
        dem_rgb = dem.repeat(1,3,1,1)

        feats = self.vgg_enc.extract_features_at_res(
                    dem_rgb, self.cfg.feature_channels.keys(), mode=self.cfg.vgg_mode)

        # --- sample t / construct noisy input -------------------
        t  = self.sample(len(dem)).to(self.device)
        w  = self.weight(t)
        eps = torch.randn_like(sat)

        loss_kind = self.cfg.loss.lower()
        if loss_kind in ("flow","fm"):
            z_t    = (1-w)[:,None,None,None]*sat + w[:,None,None,None]*eps
            target = sat - eps
        else:
            sqrt_w     = w.sqrt()[:,None,None,None]
            sqrt_one_w = (1-w).sqrt()[:,None,None,None]
            z_t    = sqrt_w*sat + sqrt_one_w*eps
            target = eps if loss_kind.startswith("eps") else sat

        text_use = torch.zeros_like(text) \
                   if random.random() < self.cfg.cfg_drop_prob else text

        # --- forward / backward --------------------------------
        self.opt.zero_grad(set_to_none=True)
        with self._amp():
            pred  = self.model(z_t, t, text_use, feats)
            loss  = self.loss_fn(pred, target)

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        if self.scaler.is_enabled():
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

        # --- EMA ------------------------------------------------
        with torch.no_grad():
            for p_ema, p in zip(self.ema.parameters(), self.model.parameters()):
                p_ema.mul_(self.cfg.ema_decay).add_(p, alpha=1-self.cfg.ema_decay)

        wandb.log({"step_loss": loss.detach().item()})

    # ----------------------------------------------------------- #
    def train(self):
        for epoch in range(self.cfg.epochs):
            self.model.train()
            for dem, sat, txt in tqdm(self.tr_loader):
                self._training_step(dem, sat, txt)

            if (epoch+1) % self.cfg.val_every == 0:
                self.validate(epoch)
            if (epoch+1) % self.cfg.save_every == 0:
                self.save(epoch)

    # ============================================================== VALIDATION
    @staticmethod
    def _ddim_step(x_t, eps_hat, w_t, w_prev):
        sqrt_w, sqrt_one_w   = w_t.sqrt(), (1-w_t).sqrt()
        sqrt_wp, sqrt_one_wp = w_prev.sqrt(), (1-w_prev).sqrt()
        x0_hat = (x_t - sqrt_one_w[...,None,None]*eps_hat) / sqrt_w[...,None,None]
        return (sqrt_wp[...,None,None]*x0_hat + sqrt_one_wp[...,None,None]*eps_hat).clamp(-1,1)

    @staticmethod
    def _flow_euler(x_t, vf_hat, dt):
        return (x_t + dt*vf_hat)

    @torch.no_grad()
    def validate(self, epoch:int):
        self.ema.eval()
        dem, _, txt = next(iter(self.va_loader))
        dem_rgb = dem.to(self.device).repeat(1,3,1,1)
        txt     = txt.to(self.device)
        feats   = self.vgg_enc.extract_features_at_res(
                    dem_rgb, self.cfg.feature_channels.keys(), mode=self.cfg.vgg_mode)
        B = dem_rgb.size(0)

        steps = 100
        t_seq = torch.linspace(1,0,steps+1,device=self.device)
        w_seq = self.weight(t_seq)
        kind  = self.cfg.loss.lower()

        for scale in (1.0,5.0,8.0):
            xt = torch.randn_like(dem_rgb)
            for s in range(steps):
                t_curr = t_seq[s].expand(B)
                t_next = t_seq[s+1].expand(B)

                with self._amp():
                    cond   = self.ema(xt, t_curr, txt, feats)
                    uncond = self.ema(xt, t_curr, torch.zeros_like(txt), feats)
                    guided = uncond + scale*(cond-uncond)

                if kind in ("flow","fm"):
                    dt = (t_curr - t_next).view(-1,1,1,1)
                    xt = self._flow_euler(xt, guided, dt)
                else:
                    eps_hat = guided if kind.startswith("eps") else \
                              (xt - w_seq[s].sqrt()*guided) / (1-w_seq[s]).sqrt()
                    xt = self._ddim_step(xt, eps_hat, w_seq[s], w_seq[s+1])

            grid  = torchvision.utils.make_grid(xt.clamp(-1,1).cpu(),
                                                nrow=8, normalize=True)
            wandb.log({f"generated_images_cfg_{scale}":
                       wandb.Image(grid, caption=f"CFG {scale} | epoch {epoch}")})

    # ============================================================== SAVE
    def save(self, epoch:int):
        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"epoch": epoch, "ema": self.ema.state_dict(),
             "raw": self.model.state_dict()},
            Path(self.cfg.out_dir)/f"epoch{epoch:03}.pth"
        )

# ─────────────────────────────────────────────────────────────── #
def run_from_yaml(path:str="configs/default.yaml"):
    Trainer(TrainCfg.from_yaml(path)).train()
import enum, random, torch, torchvision.models as models
from terrain_diffuser.core.base import ImageEncoder
from collections.abc import Sequence

class Selection(enum.Enum):
    ALL     = "all"
    TOPK    = "topk"
    BOTTOMK = "bottomk"
    RANDOM  = "random"

class VGG16Encoder(ImageEncoder):
    _RAW_C = {64: 64, 32: 128, 16: 256, 8: 512, 4: 512}
    _LAYER_TO_RES = {3: 64, 8: 32, 15: 16, 22: 8, 29: 4}

    def __init__(self,
                 feature_channels: dict[int, int],   # e.g. {16: 16, 8: 32}
                 seed: int | None = None,            # NEW — controls randomness
                 pretrained: bool = True,
                 requires_grad: bool = False,
                 device: torch.device | str = "cpu"):
        super().__init__()

        # ---------- store k per resolution -----------
        self.k_per_res = {int(r): (None if k is None else int(k))
                          for r, k in feature_channels.items()}

        # ---------- sanity checks --------------------
        for r, k in self.k_per_res.items():
            if r not in self._RAW_C:
                raise ValueError(f"Resolution {r} not available in VGG-16")
            if k is not None and k > self._RAW_C[r]:
                raise ValueError(f"k={k} exceeds {self._RAW_C[r]} channels at {r}×{r}")

        # ---------- backbone -------------------------
        self.vgg = models.vgg16(pretrained=pretrained).features
        self.vgg.eval().to(device)
        if not requires_grad:
            for p in self.vgg.parameters():
                p.requires_grad_(False)

        # ---------- fixed random indices -------------
        g = random.Random(seed)          # local RNG; seed=None → true randomness run-to-run
        self.fixed_idx: dict[int, list[int]] = {}
        for r, k in self.k_per_res.items():
            if k is not None:            # None ⇒ keep all channels
                self.fixed_idx[r] = g.sample(range(self._RAW_C[r]), k)

    # ---------------- helpers -----------------------
    @staticmethod
    def _slice(feat, idx):
        return feat[:, idx, :, :] if isinstance(idx, slice) else feat[:, idx, :, :]

    def _select(self, feat, mode: Selection, k, res):
        if mode is Selection.ALL or k is None:
            return feat
        C = feat.size(1)
        if k > C:
            raise ValueError(f"Requested k={k}, but feature map has only C={C} channels")

        if mode is Selection.RANDOM:
            idx = self.fixed_idx[res]    # deterministic per run
        elif mode is Selection.TOPK:
            idx = slice(0, k)
        else:  # BOTTOMK
            idx = slice(C - k, C)

        return self._slice(feat, idx)

    # ---------------- public API --------------------
    @torch.no_grad()
    def extract_features_at_res(self,
                                images: torch.Tensor,
                                resolutions: Sequence[int],
                                mode: str = "all") -> dict[str, torch.Tensor]:

        wanted    = {int(r) for r in resolutions}
        mode_enum = Selection(mode)
        out       = {}
        x = images

        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            r = self._LAYER_TO_RES.get(idx)
            if r and r in wanted:
                k = self.k_per_res.get(r)          # may be None
                out[str(r)] = self._select(x.clone(), mode_enum, k, r)
                wanted.remove(r)
                if not wanted:
                    break
        return out

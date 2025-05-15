from transformers import T5Tokenizer, T5EncoderModel
import torch
from terrain_diffuser.core.base import TextEncoder

class T5TextEncoder(TextEncoder):
    def __init__(self,
                 model_name: str = "google/flan-t5-small",
                 max_length: int = 77,
                 device: torch.device | str = "cpu"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name,
                                                     model_max_length=max_length)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.eval().to(device)
        self.max_length = max_length

    # ---------- API ----------
    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenize(texts)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs).last_hidden_state          # (B,S,D)
        pad_mask = inputs["attention_mask"].bool()
        out = out.masked_fill(~pad_mask.unsqueeze(-1), 0.)
        return out.cpu()

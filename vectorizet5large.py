import json
import torch
import numpy as np
import logging
from pathlib import Path
from typing import List
from transformers import T5Tokenizer, T5EncoderModel
from einops import rearrange
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

MAX_LENGTH = 128  
T5_MODEL = "google/flan-t5-large" 

try:
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL, model_max_length=MAX_LENGTH)
    model = T5EncoderModel.from_pretrained(T5_MODEL)
    logging.info("Tokenizer and model loaded successfully.\n")
except Exception as e:
    logging.error(f"Error loading tokenizer/model: {e}")
    raise e

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def tokenize(texts: List[str]):
    device = next(model.parameters()).device
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True
    ).to(device)
    return inputs, inputs["attention_mask"]

def encode_tokenized_text(token_ids, attn_mask=None, pad_id=None):
    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())
    model.eval()
    with torch.no_grad():
        output = model(input_ids=token_ids, attention_mask=attn_mask)
        output = output.last_hidden_state.detach()
    attn_mask = attn_mask.bool()
    output = output.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.)
    return output

def encode_text(texts: List[str], return_attn_mask=False):
    inputs, attn_mask = tokenize(texts)
    output = encode_tokenized_text(
        inputs["input_ids"],
        attn_mask=attn_mask,
        pad_id=tokenizer.pad_token_id
    )
    if return_attn_mask:
        return output, attn_mask.bool()
    return output

def encode_json_labels_pytorch(input_json_path: str, output_pt_path: str, batch_size: int = 32, max_samples: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device}\n")

    logging.info(f"Loading input JSON from {input_json_path}...")
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} entries from the input JSON.\n")
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}")
        raise e

    if max_samples is not None:
        data = data[:max_samples]
        logging.info(f"Processing first {max_samples} samples.\n")

    descriptions = [entry.get("description", "") for entry in data]
    image_paths = [entry.get("image_path", "") for entry in data]

    logging.info("Starting text encoding...\n")
    all_embeddings = []
    all_masks = []

    for i in tqdm(range(0, len(descriptions), batch_size), desc="Encoding descriptions"):
        batch_texts = descriptions[i:i+batch_size]
        try:
            text_embeds, attn_masks = encode_text(batch_texts, return_attn_mask=True)
            all_embeddings.append(text_embeds.cpu())
            all_masks.append(attn_masks.cpu())
        except Exception as e:
            logging.error(f"Error encoding batch {i}: {e}")
            hidden_dim = model.config.d_model
            batch_size_failed = len(batch_texts)
            failed_embeds = torch.zeros((batch_size_failed, MAX_LENGTH, hidden_dim))
            failed_masks = torch.zeros((batch_size_failed, MAX_LENGTH), dtype=torch.bool)
            all_embeddings.append(failed_embeds)
            all_masks.append(failed_masks)

    logging.info("\nConcatenating embeddings and masks...")
    try:
        text_embeds = torch.cat(all_embeddings, dim=0)
        attn_masks = torch.cat(all_masks, dim=0)
        logging.info(f"Embeddings shape: {text_embeds.shape}, Masks shape: {attn_masks.shape}\n")
    except Exception as e:
        logging.error(f"Concatenation error: {e}")
        raise e

    if len(image_paths) != text_embeds.shape[0] or len(image_paths) != attn_masks.shape[0]:
        logging.error("Mismatch between image paths and embeddings/masks count.")
        raise ValueError("Data length mismatch.")

    data_to_save = {
        "image_paths": image_paths,
        "text_embeddings": text_embeds,
        "attention_masks": attn_masks
    }

    try:
        Path(output_pt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_to_save, output_pt_path)
        logging.info(f"Saved data to {output_pt_path}")
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        raise e


# text embedding script
if __name__ == "__main__":
    input_json = "labels.json"
    output_pt = "embeddings_flant5large.pt"
    encode_json_labels_pytorch(input_json, output_pt, batch_size=8)

"""Hugging Face causal LM + PEFT LoRA. Needs transformers, peft, datasets (and bitsandbytes for 4-bit).

cfg (all optional except model and dataset):
  model             HF model id, e.g. "Qwen/Qwen2.5-1.5B"
  revision          model revision (pin it: every replica must load the same base weights)
  dataset           HF dataset id, or "json" / "text" / "parquet" with data_files
  dataset_config    dataset config name
  data_files        for local/bundled files, e.g. "data/train.jsonl" (relative to the bundle)
  split             "train"
  text_field        "text"; or set messages_field for chat data rendered with the tokenizer's chat template
  messages_field    e.g. "messages"
  seq_len           1024
  micro_batch       4 (per GPU)
  grad_accum        4
  lr                2e-4,  warmup 20,  weight_decay 0.0,  clip 1.0
  lora              {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"}
  load_in_4bit      false (QLoRA: needs bitsandbytes; the base is frozen, so DiLoCo only moves the adapter)
  gradient_checkpointing  true
  dtype             "float16" (T4 has no bfloat16)
  hf_token_env      env var holding an HF token for gated models (default HF_TOKEN)
"""
import os

import torch

_cache = {}


def _tokenizer(cfg):
    from transformers import AutoTokenizer
    key = (cfg["model"], cfg.get("revision"))
    if key not in _cache:
        tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"),
                                            token=os.environ.get(cfg.get("hf_token_env", "HF_TOKEN")))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _cache[key] = tok
    return _cache[key]


def load_model(cfg, device):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    dtype = getattr(torch, cfg.get("dtype", "float16")) if str(device).startswith("cuda") else torch.float32
    import transformers
    newer = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit()) >= (4, 56)
    kw = {"revision": cfg.get("revision"), ("dtype" if newer else "torch_dtype"): dtype,
          "token": os.environ.get(cfg.get("hf_token_env", "HF_TOKEN"))}
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    if str(device).startswith("cuda"):
        kw["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kw)
    if cfg.get("load_in_4bit"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
    elif cfg.get("gradient_checkpointing", True) and str(device).startswith("cuda"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lc = {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear", **(cfg.get("lora") or {})}
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    for p in model.parameters():  # fp32 master weights for the adapter; autocast does fp16 compute
        if p.requires_grad:
            p.data = p.data.float()
    if not str(device).startswith("cuda"):
        model.to(device)
    return model


def _dataset(cfg, shard, num_shards, seed):
    from datasets import load_dataset
    kw = {"split": cfg.get("split", "train")}
    if cfg.get("data_files"):
        kw["data_files"] = cfg["data_files"]
    ds = load_dataset(cfg["dataset"], cfg.get("dataset_config"), **kw)
    ds = ds.shuffle(seed=seed)  # same order on every machine, then disjoint shards per replica and rank
    if num_shards > 1:
        if len(ds) < num_shards:
            raise ValueError(f"dataset has {len(ds)} rows, fewer than {num_shards} shards")
        ds = ds.shard(num_shards=num_shards, index=shard, contiguous=True)
    return ds


def _text(cfg, tok, row):
    if cfg.get("messages_field"):
        return tok.apply_chat_template(row[cfg["messages_field"]], tokenize=False)
    return row[cfg.get("text_field", "text")]


def batches(cfg, shard, num_shards, start, seed):
    tok = _tokenizer(cfg)
    ds = _dataset(cfg, shard, num_shards, seed)
    mb, n = int(cfg.get("micro_batch", 4)), len(ds)
    j = start
    while True:  # wraps around: more rounds than data = more epochs
        rows = [ds[(j * mb + i) % n] for i in range(mb)]
        enc = tok([_text(cfg, tok, r) for r in rows], max_length=int(cfg.get("seq_len", 1024)), truncation=True,
                  padding="longest", return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        yield {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}
        j += 1


def loss(model, batch, device):
    return model(**{k: v.to(device) for k, v in batch.items()}).loss


def batch_tokens(batch):
    return int(batch["attention_mask"].sum())


def export(cfg, model, out_dir):
    """A standard PEFT adapter directory (adapter_config.json + adapter_model.safetensors), loadable with
    PeftModel.from_pretrained(base, out_dir), plus the tokenizer."""
    model.save_pretrained(out_dir)
    _tokenizer(cfg).save_pretrained(out_dir)

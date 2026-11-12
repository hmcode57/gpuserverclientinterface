"""Recipes tell the DiLoCo trainer what to train. A recipe is a module with:

  load_model(cfg, device) -> torch.nn.Module
      Base model frozen, LoRA (or any small set of) parameters trainable (requires_grad=True), trainable
      parameters in fp32. Must build the same parameter names and shapes on every replica.
  batches(cfg, shard, num_shards, start, seed) -> iterator
      Infinite iterator of micro-batches for data shard `shard` of `num_shards`, starting at micro-batch
      `start` (so round r of replica k always sees the same data, whichever session runs it).
  loss(model, batch, device) -> scalar tensor
  batch_tokens(batch) -> int                       (optional, for metrics and token-weighted averaging)
  export(cfg, model, out_dir)                      (optional: write a usable artifact, e.g. a PEFT adapter)

Built in: diloco.recipes.hf_lora (Hugging Face causal LM + PEFT LoRA), diloco.recipes.toy (CPU tests).
Your own recipe: put the module in the bundle (laptop/make_bundle.py --include) and pass --recipe mymod.
"""

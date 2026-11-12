"""Tiny pure-torch LoRA problem for CPU tests: learn a low-rank correction to a frozen linear map.
cfg: dim (16), rank (4), micro_batch (32), lr, seed (7), fail_at ([replica, round]: that train job fails)."""
import torch


class ToyLoRA(torch.nn.Module):
    def __init__(self, dim, rank, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = torch.nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.base.weight.copy_(torch.randn(dim, dim, generator=g) / dim ** 0.5)
        self.base.weight.requires_grad_(False)
        self.lora_A = torch.nn.Parameter(torch.randn(rank, dim) * 0.1)
        self.lora_B = torch.nn.Parameter(torch.zeros(dim, rank))

    def forward(self, x):
        return self.base(x) + x @ self.lora_A.t() @ self.lora_B.t()


def _target(cfg):
    dim, rank, seed = cfg.get("dim", 16), cfg.get("rank", 4), cfg.get("seed", 7)
    g = torch.Generator().manual_seed(seed + 1)
    base = ToyLoRA(dim, rank, seed).base.weight
    return base + torch.randn(dim, rank, generator=g) @ torch.randn(rank, dim, generator=g) / dim ** 0.5


def load_model(cfg, device):
    return ToyLoRA(cfg.get("dim", 16), cfg.get("rank", 4), cfg.get("seed", 7)).to(device)


def batches(cfg, shard, num_shards, start, seed):
    w = _target(cfg)
    import os
    if list(cfg.get("fail_at") or []) == [int(os.environ.get("DILOCO_REPLICA", -1)), int(os.environ.get("DILOCO_ROUND", -1))]:
        raise RuntimeError("toy recipe asked to fail (fail_at)")
    slow = cfg.get("sleep_at")  # [replica, round, seconds]: that train job stalls (straggler tests)
    if slow and slow[:2] == [int(os.environ.get("DILOCO_REPLICA", -1)), int(os.environ.get("DILOCO_ROUND", -1))]:
        import time
        time.sleep(slow[2])
    j = start
    while True:
        g = torch.Generator().manual_seed(hash((seed, shard, num_shards, j)) % 2**31)
        x = torch.randn(cfg.get("micro_batch", 32), w.shape[0], generator=g)
        yield {"x": x, "y": x @ w.t()}
        j += 1


def loss(model, batch, device):
    return torch.nn.functional.mse_loss(model(batch["x"].to(device)), batch["y"].to(device))


def batch_tokens(batch):
    return batch["x"].shape[0]


def eval_loss(cfg, state, n=512):
    """Held-out loss of a trainable state (tests)."""
    m = load_model(cfg, "cpu")
    with torch.no_grad():
        m.lora_A.copy_(state["lora_A"]); m.lora_B.copy_(state["lora_B"])
        g = torch.Generator().manual_seed(12345)
        x = torch.randn(n, cfg.get("dim", 16), generator=g)
        return float(torch.nn.functional.mse_loss(m(x), x @ _target(cfg).t()))

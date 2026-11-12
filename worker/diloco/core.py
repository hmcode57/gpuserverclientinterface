"""DiLoCo math (needs torch): trainable-state extraction, int8 pseudo-gradients, the outer Nesterov step.

Notation (Douillard et al. 2023, "DiLoCo: Distributed Low-Communication Training of Language Models"):
  theta      global trainable weights (here: the LoRA adapter), held as fp32
  delta_k    pseudo-gradient of replica k = theta - theta_k after H inner steps
  outer step g = mean_k delta_k;  v = mu * v + g;  theta = theta - lr * (g + mu * v)   (Nesterov SGD)
With one replica, lr=1 and mu=0 the outer step returns exactly the replica's weights (plain training).
"""
import io

import torch


def trainable_state(model):
    """{name: fp32 CPU copy} of every parameter with requires_grad (the LoRA weights)."""
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, state):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    missing, extra = set(params) - set(state), set(state) - set(params)
    if missing or extra:
        raise ValueError(f"state does not match the model's trainable parameters: missing {sorted(missing)[:5]}, "
                         f"unexpected {sorted(extra)[:5]} (same recipe and cfg on every replica?)")
    with torch.no_grad():
        for n, p in params.items():
            if p.shape != state[n].shape:
                raise ValueError(f"{n}: shape {tuple(state[n].shape)} != model {tuple(p.shape)}")
            p.copy_(state[n].to(p.device, p.dtype))


def quantize(t):
    """int8 with one fp32 absmax scale per row (per tensor for vectors)."""
    flat = t.float().reshape(t.shape[0] if t.ndim >= 2 else 1, -1)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    q = torch.round(flat / scale).clamp_(-127, 127).to(torch.int8)
    return {"q": q, "scale": scale.squeeze(1), "shape": list(t.shape)}


def dequantize(d):
    return (d["q"].float() * d["scale"].unsqueeze(1)).reshape(d["shape"])


def make_delta(theta, local, bits=8):
    """Pseudo-gradient theta - local, int8-quantized unless bits=32."""
    out = {}
    for n, t in theta.items():
        d = t.float() - local[n].float()
        out[n] = quantize(d) if bits == 8 else {"f": d}
    return {"format": f"delta-{bits}", "tensors": out}


def read_delta(obj):
    return {n: dequantize(d) if "q" in d else d["f"] for n, d in obj["tensors"].items()}


def outer_step(theta, deltas, momentum=None, lr=0.7, mu=0.9, weights=None):
    """Returns (new_theta, new_momentum, stats). `deltas` is a list of {name: tensor}; `weights` optional
    per-replica weights (e.g. tokens seen), normalized to sum 1."""
    if not deltas:
        raise ValueError("no deltas to average")
    w = torch.tensor(weights or [1.0] * len(deltas), dtype=torch.float64)
    w = (w / w.sum()).tolist()
    new_theta, new_mom = {}, {}
    sq = 0.0
    for n, t in theta.items():
        g = sum(wi * d[n].float() for wi, d in zip(w, deltas))
        sq += float(g.pow(2).sum())
        v = g.clone() if momentum is None or mu == 0 else mu * momentum[n].float() + g
        new_mom[n] = v
        new_theta[n] = t.float() - lr * (g + mu * v if mu else g)
    return new_theta, new_mom, {"pseudo_grad_norm": sq ** 0.5, "replicas": len(deltas)}


def to_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def from_bytes(b):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


def save(obj, path):
    torch.save(obj, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)

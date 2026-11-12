"""DiLoCo work that needs torch, run by the handler as a subprocess (under torchrun for multi-GPU training):

  python3 worker/diloco/inner.py SPEC.json                                         # init / reduce / export / 1-GPU train
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 worker/diloco/inner.py SPEC.json   # train on 2 GPUs

SPEC.json: {"op": "init"|"train"|"reduce"|"export", "work": dir, ...}. Inputs are files in `work` that the handler
downloaded; outputs are written to `work` for the handler to upload. Progress goes to work/progress.json, the
result to work/result.json. Recipe interface: see worker/diloco/recipes/__init__.py.
"""
import importlib, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]  # worker/ (so `diloco` imports), plus the cwd for user recipes
sys.path.append(os.getcwd())

import torch  # noqa: E402

from diloco import core  # noqa: E402


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def recipe(spec):
    return importlib.import_module(spec["recipe"])


def lr_factor(step, warmup, total, min_ratio=0.1):
    """Linear warmup, then cosine to min_ratio over the whole run (all rounds), on the global inner step."""
    if warmup and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def one_device(spec):
    """init/export: the first GPU if there is one (a 7B base in fp32 would not fit in a notebook's RAM)."""
    return "cuda:0" if torch.cuda.is_available() and not spec.get("cpu") else "cpu"


def op_init(spec, w):
    torch.manual_seed(spec.get("seed", 0))
    model = recipe(spec).load_model(spec["cfg"], one_device(spec))
    state = core.trainable_state(model)
    core.save(state, os.path.join(w, "theta.pt"))
    n = sum(t.numel() for t in state.values())
    return {"trainable_params": n, "tensors": len(state)}


def op_reduce(spec, w):
    theta = core.load(os.path.join(w, "theta.pt"))
    mom_path = os.path.join(w, "momentum.pt")
    momentum = core.load(mom_path) if os.path.exists(mom_path) else None
    deltas, weights = [], []
    for i, name in enumerate(spec["deltas"]):
        deltas.append(core.read_delta(core.load(os.path.join(w, name))))
        weights.append(float((spec.get("weights") or [1.0] * len(spec["deltas"]))[i]))
    new_theta, new_mom, stats = core.outer_step(theta, deltas, momentum, spec.get("outer_lr", 0.7),
                                                spec.get("outer_momentum", 0.9), weights)
    core.save(new_theta, os.path.join(w, "theta_next.pt"))
    core.save(new_mom, os.path.join(w, "momentum_next.pt"))
    return stats


def op_export(spec, w):
    r = recipe(spec)
    model = r.load_model(spec["cfg"], one_device(spec))
    core.load_trainable_state(model, core.load(os.path.join(w, "theta.pt")))
    out = os.path.join(w, "export")
    os.makedirs(out, exist_ok=True)
    if hasattr(r, "export"):
        r.export(spec["cfg"], model, out)
    else:
        core.save(core.trainable_state(model), os.path.join(out, "trainable_state.pt"))
    return {"files": sorted(os.listdir(out))}


def op_train(spec, w):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = torch.cuda.is_available() and not spec.get("cpu")
    if cuda:
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if cuda else "cpu"
    if world > 1:
        torch.distributed.init_process_group("nccl" if cuda else "gloo")
    cfg, r = spec["cfg"], recipe(spec)
    os.environ["DILOCO_REPLICA"], os.environ["DILOCO_ROUND"] = str(spec["replica"]), str(spec["round"])
    torch.manual_seed(spec.get("seed", 0) * 1000 + spec["replica"] * 10 + rank)

    model = r.load_model(cfg, device)
    theta = core.load(os.path.join(w, "theta.pt"))
    core.load_trainable_state(model, theta)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-4), weight_decay=cfg.get("weight_decay", 0.0),
                            betas=tuple(cfg.get("betas", (0.9, 0.95))))
    state_path = os.path.join(w, "inner_state.pt")
    resumed = os.path.exists(state_path)
    if resumed:  # this replica's AdamW state from its last good round (DiLoCo keeps inner state per replica)
        opt.load_state_dict(core.load(state_path))
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if cuda else None) \
        if world > 1 else model
    fp16 = cuda and cfg.get("amp", "fp16") == "fp16"  # T4 has no bf16
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    steps, accum = int(spec["steps"]), int(cfg.get("grad_accum", 1))
    base_lr, warmup, total = cfg.get("lr", 2e-4), int(cfg.get("warmup", 0)), int(spec["total_steps"])
    start_step = int(spec["round"]) * steps
    shards = int(spec["num_replicas"]) * world
    batches = r.batches(cfg, shard=int(spec["replica"]) * world + rank, num_shards=shards,
                        start=start_step * accum, seed=spec.get("seed", 0))
    losses, t0, tokens = [], time.time(), 0
    model.train()
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = base_lr * lr_factor(start_step + i, warmup, total)
        total_loss = 0.0
        for m in range(accum):
            batch = next(batches)
            sync = m == accum - 1
            ctx = net.no_sync() if world > 1 and not sync else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = r.loss(net, batch, device) / accum
            scaler.scale(loss).backward()
            total_loss += float(loss.detach())
            tokens += int(r.batch_tokens(batch)) if hasattr(r, "batch_tokens") else 0
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0))
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if world > 1:  # mean loss across ranks, for logging only
            t = torch.tensor([total_loss], device=device)
            torch.distributed.all_reduce(t)
            total_loss = float(t) / world
        losses.append(total_loss)
        if rank == 0 and (i % 5 == 0 or i == steps - 1):
            write_json(os.path.join(w, "progress.json"),
                       {"step": i + 1, "steps": steps, "loss": round(total_loss, 5), "lr": opt.param_groups[0]["lr"]})
    if world > 1:
        t = torch.tensor([tokens], device=device)
        torch.distributed.all_reduce(t)
        tokens = int(t)
    result = None
    if rank == 0:
        local = core.trainable_state(model)
        delta = core.make_delta(theta, local, bits=int(spec.get("delta_bits", 8)))
        core.save(delta, os.path.join(w, "delta.pt"))
        if spec.get("keep_inner_state", True):
            core.save(opt.state_dict(), os.path.join(w, "inner_state_next.pt"))
        k = max(1, len(losses) // 10)
        result = {"loss_first": round(sum(losses[:k]) / k, 5), "loss_last": round(sum(losses[-k:]) / k, 5),
                  "steps": steps, "world": world, "tokens": tokens, "train_s": round(time.time() - t0, 1),
                  "inner_state_resumed": resumed}
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return result


def main():
    spec = json.load(open(sys.argv[1]))
    w = spec["work"]
    fn = {"init": op_init, "train": op_train, "reduce": op_reduce, "export": op_export}[spec["op"]]
    result = fn(spec, w)
    if result is not None and int(os.environ.get("RANK", "0")) == 0:
        write_json(os.path.join(w, "result.json"), result)


if __name__ == "__main__":
    main()

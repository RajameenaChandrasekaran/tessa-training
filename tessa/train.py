"""TESSA pre-training (Variant B). Launch with torchrun, one process per GPU:

  torchrun --standalone --nproc_per_node=3 -m tessa.train --eo_dir ... --text_dir ... --sar_pack ... --out_dir ...

See scripts/launch_train_3gpu.sh for the full recommended command.
"""
import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from . import paths as P
from .data import CudaPrefetcher, TessaDataset, sar_to_float, usable_rows
from .split import load_split, make_split, report as split_report, save_split
from .model import TessaB


# ----------------------------------------------------------------------------------------------- args
def get_args(argv=None):
    p = argparse.ArgumentParser("TESSA Variant B pre-training")
    g = p.add_argument_group("data")
    g.add_argument("--eo_dir", help="cache of DINOv3 tokens (tools/cache_dinov3_tokens.py)")
    g.add_argument("--text_dir", help="cache of Qwen text vectors (tools/cache_text_embeddings.py)")
    g.add_argument("--sar_pack", help="packed uint8 SAR tiles (tools/pack_sar_uint8.py); recommended for speed")
    g.add_argument("--sar_dir", default=P.DEFAULT_SAR_DIR, help="SAR tiles read directly if no --sar_pack (default from tessa/paths.py)")
    g.add_argument("--paths_json", default=P.DEFAULT_PATHS_JSON, help="tile -> file index from tools/check_dataset_paths.py")
    g.add_argument("--sar_pattern", default="{tile}")
    g.add_argument("--image_size", type=int, default=512)
    g.add_argument("--split_file", default="", help="train/val/test split (created if missing; default <out_dir>/split.json)")
    g.add_argument("--val_n", type=int, default=4096, help="validation tiles (monitoring only)")
    g.add_argument("--test_n", type=int, default=4096, help="test tiles (held out, never used by this script)")
    g.add_argument("--split_seed", type=int, default=0)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--limit", type=int, default=0, help="use only the first N tiles (debug)")
    g = p.add_argument_group("model")
    g.add_argument("--patch", type=int, default=16)
    g.add_argument("--embed_dim", type=int, default=768); g.add_argument("--depth", type=int, default=12)
    g.add_argument("--heads", type=int, default=12)
    g.add_argument("--pred_dim", type=int, default=384); g.add_argument("--pred_depth", type=int, default=6)
    g.add_argument("--pred_heads", type=int, default=6)
    g.add_argument("--eo_vis_head", default="none", choices=["none", "linear", "mlp"])
    g.add_argument("--eo_mask_head", default="linear", choices=["linear", "mlp"])
    g.add_argument("--txt_hidden", type=int, default=1024)
    g.add_argument("--mask_ratio", type=float, default=0.75)
    g.add_argument("--mgf_impl", default="fast", choices=["fast", "conv"])
    g.add_argument("--norm_target", action="store_true", help="per-patch standardised structure targets (SARATR-X style)")
    g = p.add_argument_group("loss")
    g.add_argument("--lambda_vis", type=float, default=0.1); g.add_argument("--lambda_mask", type=float, default=0.1)
    g.add_argument("--lambda_txt", type=float, default=0.1)
    g.add_argument("--ramp_epochs", type=float, default=10.0, help="linear ramp of the three auxiliary weights from 0")
    g.add_argument("--dup_cos", type=float, default=None, help="also treat text pairs with cosine above this as positives")
    g = p.add_argument_group("optim")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch_size", type=int, default=256, help="per GPU")
    g.add_argument("--accum_iter", type=int, default=1)
    g.add_argument("--blr", type=float, default=3e-4, help="lr = blr * effective_batch / 256")
    g.add_argument("--min_lr", type=float, default=0.0)
    g.add_argument("--warmup_epochs", type=float, default=10.0)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--clip_grad", type=float, default=1.0)
    g.add_argument("--contrast", type=float, default=0.5, help="contrast jitter (SAR-JEPA uses 0.5); 0 disables")
    g = p.add_argument_group("system")
    g.add_argument("--amp", default="bf16", choices=["bf16", "fp32"])
    g.add_argument("--workers", type=int, default=12, help="dataloader workers per GPU")
    g.add_argument("--prefetch", type=int, default=4)
    g.add_argument("--compile", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--log_every", type=int, default=20)
    g.add_argument("--save_every", type=int, default=10)
    g.add_argument("--val_every", type=int, default=5)
    g.add_argument("--val_batch", type=int, default=64)
    g.add_argument("--resume", default="auto", help="'auto' (use ckpt_last.pt if present), a path, or ''")
    g.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    g.add_argument("--grad_log_every", type=int, default=0, help="log per-loss grad norms on the last encoder block")
    g.add_argument("--no_save", action="store_true", help="skip checkpoints (benchmarks / smoke tests)")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------- helpers
def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    return rank, world, local, device


def lr_at(epoch_f, a, base_lr):
    if epoch_f < a.warmup_epochs:
        return base_lr * epoch_f / max(a.warmup_epochs, 1e-8)
    t = (epoch_f - a.warmup_epochs) / max(a.epochs - a.warmup_epochs, 1e-8)
    return a.min_lr + (base_lr - a.min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def gather_text(txt, grp, world):
    if world == 1:
        return txt, grp
    tl = [torch.empty_like(txt) for _ in range(world)]
    gl = [torch.empty_like(grp) for _ in range(world)]
    dist.all_gather(tl, txt.contiguous())
    dist.all_gather(gl, grp.contiguous())
    return torch.cat(tl), torch.cat(gl)


def effective_rank(x):
    x = x.float() - x.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def retrieval_metrics(z, t, g):
    """z: image embeddings, t: text targets (both L2-normalised, [N,D]); g: caption-group ids.
    A text/image counts as correct if it is in the same caption group, and other positives never count against a query."""
    sim = z.float() @ t.float().T
    pos = g[:, None] == g[None, :]
    out = {}
    best = sim.masked_fill(~pos, float("-inf")).max(1).values
    rank_i2t = ((sim > best[:, None]) & ~pos).sum(1) + 1
    best_t = sim.masked_fill(~pos, float("-inf")).max(0).values
    rank_t2i = ((sim > best_t[None, :]) & ~pos).sum(0) + 1
    for k in (1, 5, 10):
        out[f"i2t_r{k}"] = (rank_i2t <= k).float().mean().item()
        out[f"t2i_r{k}"] = (rank_t2i <= k).float().mean().item()
    out["chance_r1"] = (pos.float().sum(1).mean() / pos.shape[1]).item()
    return out


def save_ckpt(path, model, opt, epoch, args, extra=None):
    tmp = str(path) + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "args": vars(args),
                **(extra or {})}, tmp)
    os.replace(tmp, path)


def autocast_ctx(args, device):
    if args.amp == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, val_ds, args, device, flags):
    model.eval()
    loader = DataLoader(val_ds, batch_size=args.val_batch, shuffle=False, num_workers=min(args.workers, 8),
                        pin_memory=device.type == "cuda")
    torch.manual_seed(args.seed + 999)
    sums, n = {}, 0
    zs, ts, gs, pools = [], [], [], []
    for b in loader:
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        imgs = sar_to_float(b["sar"], 0.0)
        with autocast_ctx(args, device):
            out = model(imgs, b.get("eo"), b.get("eo_ok"), b.get("txt"), b.get("grp"), b.get("grp"), b.get("txt"), **flags)
            z, pooled = model.embed_global(imgs)
        bs = imgs.shape[0]
        for k in ("struct", "eo_vis", "eo_mask", "txt", "eo_cos_vis", "eo_cos_mask", "txt_acc"):
            if k in out:
                sums[k] = sums.get(k, 0.0) + float(out[k]) * bs
        n += bs
        zs.append(z.float()); pools.append(pooled)
        if "txt" in b:
            ts.append(b["txt"].float()); gs.append(b["grp"])
    res = {f"val_{k}": v / n for k, v in sums.items()}
    if ts:
        res.update({f"val_{k}": v for k, v in retrieval_metrics(torch.cat(zs), torch.cat(ts), torch.cat(gs)).items()})
    res["val_eff_rank"] = effective_rank(torch.cat(pools))
    model.train()
    return res


def loss_grad_norms(model, out, keys):
    params = [p for p in model.blocks[-1].parameters()]
    res = {}
    for k in keys:
        if not out[k].requires_grad:
            continue
        gr = torch.autograd.grad(out[k], params, retain_graph=True, allow_unused=True)
        res[f"gnorm_{k}"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gr if g is not None)))
    return res


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    a = get_args(argv)
    rank, world, local, device = setup_dist()
    is_main = rank == 0
    out_dir = Path(a.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(a), indent=2))
    if is_main and not a.no_save:
        import shutil
        free = shutil.disk_usage(out_dir).free / 2 ** 30
        need = 1.5 + 0.8 * (a.epochs // max(a.save_every, 1) + 1)
        print(f"[tessa] free disk at out_dir: {free:.1f} GiB (checkpoints need roughly {need:.0f} GiB over the run)", flush=True)
        if free < need:
            raise SystemExit(f"not enough free disk space for checkpoints ({free:.1f} < {need:.0f} GiB); free space or use --no_save")
    if world > 1:
        dist.barrier()
    torch.manual_seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ---- data
    full = TessaDataset(np.arange(1), a.eo_dir, a.text_dir, a.sar_pack, a.sar_dir, a.sar_pattern, a.image_size, a.paths_json)
    n_all = len(full.tiles) if not a.limit else min(a.limit, len(full.tiles))
    tiles_used = full.tiles[:n_all]
    split_path = Path(a.split_file) if a.split_file else out_dir / "split.json"
    if split_path.exists():
        sp = load_split(str(split_path), tiles_used)
    else:
        sp = make_split(tiles_used, a.val_n, a.test_n, a.split_seed)
        if is_main:
            save_split(str(split_path), sp)
    ok_rows = usable_rows(a.eo_dir, a.sar_pack, len(full.tiles))
    train_idx, val_idx = (np.array([i for i in sp[k] if ok_rows[i]], dtype=np.int64) for k in ("train", "val"))
    if is_main:
        dropped = sum(len(sp[k]) for k in ("train", "val")) - len(train_idx) - len(val_idx)
        print(f"[split] {split_path}: train {len(train_idx)} | val {len(val_idx)} | test {len(sp['test'])} (test is never loaded here)"
              f"{f' | dropped {dropped} unusable rows' if dropped else ''}", flush=True)
        if len(tiles_used) > 1000:
            print(split_report(tiles_used, sp), flush=True)
    mk = lambda idx: TessaDataset(idx, a.eo_dir, a.text_dir, a.sar_pack, a.sar_dir, a.sar_pattern, a.image_size, a.paths_json)
    train_ds, val_ds = mk(train_idx), mk(val_idx)
    eo_dim = full.eo_meta["dim"] if a.eo_dir else 768
    txt_dim = int(np.load(Path(a.text_dir) / "text.npy", mmap_mode="r").shape[1]) if a.text_dir else 8
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=a.seed, drop_last=True)
    loader = DataLoader(train_ds, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                        pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=a.workers > 0, prefetch_factor=a.prefetch if a.workers > 0 else None)
    prefetch = CudaPrefetcher(loader, device)

    # ---- model
    model = TessaB(img_size=a.image_size, patch=a.patch, embed_dim=a.embed_dim, depth=a.depth, heads=a.heads,
                   pred_dim=a.pred_dim, pred_depth=a.pred_depth, pred_heads=a.pred_heads, eo_dim=eo_dim,
                   txt_dim=txt_dim, mask_ratio=a.mask_ratio, eo_vis_head=a.eo_vis_head, eo_mask_head=a.eo_mask_head,
                   txt_hidden=a.txt_hidden, mgf_impl=a.mgf_impl, norm_target=a.norm_target).to(device)
    flags = dict(use_vis=a.lambda_vis > 0 and bool(a.eo_dir), use_mask=a.lambda_mask > 0 and bool(a.eo_dir),
                 use_txt=a.lambda_txt > 0 and bool(a.text_dir), dup_cos=a.dup_cos)
    eff_batch = a.batch_size * a.accum_iter * world
    base_lr = a.blr * eff_batch / 256
    no_decay = model.no_decay_names()
    decay = [p for n, p in model.named_parameters() if p.requires_grad and n not in no_decay]
    nodec = [p for n, p in model.named_parameters() if p.requires_grad and n in no_decay]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodec, "weight_decay": 0.0}],
                            lr=base_lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    start_epoch = 0
    ckpt_path = out_dir / "ckpt_last.pt"
    rp = ckpt_path if a.resume == "auto" and ckpt_path.exists() else (Path(a.resume) if a.resume not in ("", "auto") else None)
    if rp is not None:
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start_epoch = ck["epoch"] + 1
        if is_main:
            print(f"[resume] {rp} -> starting at epoch {start_epoch}", flush=True)

    train_model = model
    if world > 1:
        unused = not (flags["use_txt"] and flags["use_mask"] and (flags["use_vis"] or a.eo_vis_head == "none"))
        train_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None, broadcast_buffers=False,
            gradient_as_bucket_view=True, find_unused_parameters=unused)
    if a.compile:
        train_model = torch.compile(train_model)

    iters = len(loader)
    if is_main:
        n_enc = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("patch_embed", "blocks", "norm")))
        print(f"[tessa] world={world} device={device} tiles train/val={len(train_ds)}/{len(val_ds)} "
              f"batch/gpu={a.batch_size} accum={a.accum_iter} eff_batch={eff_batch} lr={base_lr:.2e} "
              f"iters/epoch={iters} grid={model.grid}x{model.grid} visible={int(round(model.L*(1-a.mask_ratio)))} "
              f"enc_params={n_enc/1e6:.1f}M total={sum(p.numel() for p in model.parameters())/1e6:.1f}M "
              f"flags={flags} amp={a.amp}", flush=True)

    # ---- train
    train_model.train()
    step = start_epoch * (iters // a.accum_iter)
    t_log, seen, data_wait = time.perf_counter(), 0, 0.0
    run = {}
    done = False
    for epoch in range(start_epoch, a.epochs):
        sampler.set_epoch(epoch)
        it = iter(prefetch)
        i = -1
        while True:
            t0 = time.perf_counter()
            batch = next(it, None)
            data_wait += time.perf_counter() - t0
            if batch is None:
                break
            i += 1
            last_micro = (i + 1) % a.accum_iter == 0
            ep_f = epoch + i / iters
            if i % a.accum_iter == 0:
                lr = lr_at(ep_f, a, base_lr)
                for gp in opt.param_groups:
                    gp["lr"] = lr
            ramp = 1.0 if a.ramp_epochs <= 0 else min(1.0, ep_f / a.ramp_epochs)
            wv, wm, wt = a.lambda_vis * ramp, a.lambda_mask * ramp, a.lambda_txt * ramp
            imgs = sar_to_float(batch["sar"], a.contrast)
            txt_all = grp_all = None
            if "txt" in batch:
                txt_all, grp_all = gather_text(batch["txt"], batch["grp"], world)
            sync = train_model.no_sync() if (world > 1 and not last_micro and not a.compile) else contextlib.nullcontext()
            with sync:
                with autocast_ctx(a, device):
                    out = train_model(imgs, batch.get("eo"), batch.get("eo_ok"), txt_all, batch.get("grp"), grp_all,
                                      batch.get("txt"), **flags)
                    loss = (out["struct"] + wv * out["eo_vis"] + wm * out["eo_mask"] + wt * out["txt"]) / a.accum_iter
                if is_main and a.grad_log_every and step % a.grad_log_every == 0 and last_micro:
                    gn = loss_grad_norms(model, out, ("struct", "eo_vis", "eo_mask", "txt"))
                    print("[gradnorm]", {k: round(v, 4) for k, v in gn.items()}, flush=True)
                loss.backward()
            for k in ("struct", "eo_vis", "eo_mask", "txt", "eo_cos_vis", "eo_cos_mask", "txt_acc"):
                if k in out:
                    run[k] = run.get(k, 0.0) + out[k].detach().float()
            run["n"] = run.get("n", 0) + 1
            seen += imgs.shape[0] * world
            if last_micro:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad) if a.clip_grad > 0 else None
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % a.log_every == 0:
                    keys = [k for k in run if k != "n"]
                    vec = torch.stack([run[k] / run["n"] for k in keys]).float()
                    if world > 1:
                        dist.all_reduce(vec); vec /= world
                    dt = time.perf_counter() - t_log
                    stats = {k: float(v) for k, v in zip(keys, vec)}
                    stats.update(epoch=epoch, step=step, lr=lr, ramp=ramp, img_s=seen / dt,
                                 data_wait_frac=data_wait / dt, gnorm=float(gnorm) if gnorm is not None else None)
                    if device.type == "cuda":
                        stats["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                    if not all(math.isfinite(v) for k, v in stats.items() if isinstance(v, float)):
                        raise RuntimeError(f"non-finite value in {stats}")
                    if is_main:
                        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()), flush=True)
                        with open(out_dir / "log.jsonl", "a") as f:
                            f.write(json.dumps(stats) + "\n")
                    run, seen, data_wait, t_log = {}, 0, 0.0, time.perf_counter()
                if a.max_steps and step >= a.max_steps:
                    done = True
                    break
        if done:
            break
        if (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs:
            if is_main:
                res = evaluate(model, val_ds, a, device, flags)
                res["epoch"] = epoch
                print("[val]", {k: round(v, 4) for k, v in res.items()}, flush=True)
                with open(out_dir / "log.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
            if world > 1:
                dist.barrier()
        if is_main and not a.no_save:
            save_ckpt(ckpt_path, model, opt, epoch, a)
            if (epoch + 1) % a.save_every == 0 or epoch + 1 == a.epochs:
                torch.save(model.encoder_state_dict(), out_dir / f"encoder_ep{epoch + 1:03d}.pt")
                torch.save({"model": model.state_dict(), "args": vars(a)}, out_dir / f"full_ep{epoch + 1:03d}.pt")
        if world > 1:
            dist.barrier()
    if is_main and not a.no_save:
        torch.save(model.encoder_state_dict(), out_dir / "encoder_final.pt")
        print("[tessa] finished", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

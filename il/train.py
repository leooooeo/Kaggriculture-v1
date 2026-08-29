"""Behavioral-cloning trainer.

    python -m il.train --shards data/shards --out models/policy.pt --epochs 30

Loads all shards into memory (fine for tens of games), trains the factored
policy, and saves weights + the channel/feature dims needed at inference.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader

from . import spec
from . import features as F
from . import action_codec as A
from .model import PolicyNet, N_BUCKETS

IGNORE = A.IGNORE
MK_KEYS = list(A.empty_market_labels().keys())


class ShardDataset(Dataset):
    def __init__(self, shard_dir):
        paths = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
        if not paths:
            raise SystemExit(f"No shards in {shard_dir}")
        parts = {k: [] for k in
                 ["board", "glob", "upos", "ulab", "umask"] +
                 [f"mk_{k}" for k in MK_KEYS]}
        for p in paths:
            z = np.load(p)
            for k in parts:
                parts[k].append(z[k])
        self.data = {k: np.concatenate(v, 0) for k, v in parts.items()}
        self.n = self.data["board"].shape[0]
        print(f"Loaded {len(paths)} shards, {self.n} steps")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.data.items()}

    def unit_type_freq(self):
        """Count of each unit action type over all valid units."""
        counts = np.ones(spec.N_UNIT_ACTIONS, np.float64)  # +1 smoothing
        lab = self.data["ulab"][..., 0]         # (N,U)
        mask = self.data["umask"].astype(bool)  # (N,U)
        vals = lab[mask]
        for v in range(spec.N_UNIT_ACTIONS):
            counts[v] += np.count_nonzero(vals == v)
        return counts

    def market_pos_rate(self, name):
        p = self.data[f"mk_{name}_present"]     # (N,S) float
        return p.mean(0)                         # per-slot positive rate


def collate(batch):
    out = {}
    for k in batch[0]:
        arr = np.stack([b[k] for b in batch], 0)
        if k in ("board", "glob") or k.endswith("_present") or k == "mk_buy_land":
            out[k] = torch.from_numpy(arr).float()
        else:
            out[k] = torch.from_numpy(arr).long()
    return out


def masked_ce(logits, target, mask, smoothing=0.02):
    """logits (N,K), target (N,), mask (N,) bool."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return Fn.cross_entropy(logits[mask], target[mask], label_smoothing=smoothing)


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ShardDataset(args.shards)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=collate, num_workers=args.workers, drop_last=True)
    net = PolicyNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(dl))

    # inverse-frequency class weights (sqrt-tempered) for the unit-type head,
    # with an extra explicit PASS down-weight; rare verbs (BUILD, PLACE...)
    # otherwise get ignored under heavy class imbalance.
    freq = ds.unit_type_freq()
    inv = (freq.sum() / (len(freq) * freq)) ** 0.5
    type_w = torch.tensor(inv, dtype=torch.float32, device=dev)
    type_w[spec.UNIT_ACTION_IDX["PASS"]] *= args.pass_weight
    type_w = (type_w / type_w.mean()).clamp(0.1, 8.0)

    # pos_weight for each market-presence BCE (orders are sparse per slot).
    mkt_pos_w = {}
    for name in ("sell", "bprod", "bseed", "banim"):
        rate = np.clip(ds.market_pos_rate(name), 1e-4, 0.999)
        mkt_pos_w[name] = torch.tensor(
            np.clip((1 - rate) / rate, 1.0, 50.0), dtype=torch.float32, device=dev)

    for ep in range(args.epochs):
        net.train()
        agg = {}
        for batch in dl:
            b = {k: v.to(dev) for k, v in batch.items()}
            (ulogits, mlogits) = net(b["board"], b["glob"], b["upos"], b["umask"])

            umask = b["umask"].bool().view(-1)                 # (B*U,)
            ulab = b["ulab"].view(-1, 4)                        # type,crop,item,count
            tp = ulab[:, 0]
            # unit type (weighted CE over valid units)
            tl = ulogits["type"].reshape(-1, spec.N_UNIT_ACTIONS)
            loss_type = Fn.cross_entropy(tl[umask], tp[umask], weight=type_w,
                                         label_smoothing=0.02)
            # crop only where PLANT
            crop_m = umask & (tp == spec.UNIT_ACTION_IDX["PLANT"]) & (ulab[:, 1] != IGNORE)
            loss_crop = masked_ce(ulogits["crop"].reshape(-1, len(spec.CROPS)),
                                  ulab[:, 1].clamp(min=0), crop_m)
            # item/count only where PICKUP/PLACE
            pp = spec.UNIT_ACTION_IDX["PICKUP"]; pl = spec.UNIT_ACTION_IDX["PLACE"]
            io_m = umask & ((tp == pp) | (tp == pl)) & (ulab[:, 2] != IGNORE)
            loss_item = masked_ce(ulogits["item"].reshape(-1, len(spec.UNIT_ITEMS)),
                                  ulab[:, 2].clamp(min=0), io_m)
            cnt_m = umask & ((tp == pp) | (tp == pl)) & (ulab[:, 3] != IGNORE)
            loss_count = masked_ce(ulogits["count"].reshape(-1, N_BUCKETS),
                                   ulab[:, 3].clamp(min=0), cnt_m)

            # ---- market ----
            loss_m = 0.0
            for name, slots in (("sell", spec.SELL_SLOTS), ("bprod", spec.BUY_PRODUCT_SLOTS),
                                ("bseed", spec.BUY_SEED_SLOTS), ("banim", spec.BUY_ANIMAL_SLOTS)):
                pres_t = b[f"mk_{name}_present"]                # (B,S) float
                pres_l = mlogits[f"{name}_present"]
                loss_m = loss_m + Fn.binary_cross_entropy_with_logits(
                    pres_l, pres_t, pos_weight=mkt_pos_w[name])
                qty_t = b[f"mk_{name}_qty"]                     # (B,S) long, IGNORE where absent
                qty_l = mlogits[f"{name}_qty"]                 # (B,S,Nb)
                qm = (qty_t != IGNORE)
                if qm.any():
                    loss_m = loss_m + Fn.cross_entropy(
                        qty_l[qm], qty_t[qm].clamp(min=0), label_smoothing=0.02)
            loss_hire = Fn.cross_entropy(mlogits["hire"], b["mk_hire"].clamp(0, spec.MAX_HIRE))
            loss_land = Fn.binary_cross_entropy_with_logits(mlogits["buy_land"], b["mk_buy_land"])

            loss = (loss_type + 0.5 * loss_crop + 0.5 * loss_item + 0.3 * loss_count
                    + args.market_weight * (loss_m + loss_hire + loss_land))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()

            for k, v in (("type", loss_type), ("crop", loss_crop), ("item", loss_item),
                         ("count", loss_count), ("mkt", loss_m), ("hire", loss_hire),
                         ("land", loss_land), ("all", loss)):
                agg[k] = agg.get(k, 0.0) + float(v.detach() if hasattr(v, "detach") else v)
        n = len(dl)
        msg = " ".join(f"{k}={agg[k]/n:.3f}" for k in ["all", "type", "crop", "item", "count", "mkt", "hire", "land"])
        print(f"ep {ep+1:2d}/{args.epochs}  {msg}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "c_board": F.C_BOARD, "g_global": F.G_GLOBAL,
                "spec_version": 1}, args.out)
    print(f"Saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--out", default="models/policy.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pass_weight", type=float, default=0.3)
    ap.add_argument("--market_weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    train(ap.parse_args())


if __name__ == "__main__":
    main()

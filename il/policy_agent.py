"""Runtime agent: obs -> action dict using the trained factored policy.

Loads weights lazily from a path next to this file (or IL_POLICY_PATH env).
If torch or the weights are unavailable it falls back to a minimal safe
heuristic so a submission never hard-errors during validation.
"""
import os
import numpy as np

from . import spec
from . import features as F
from . import action_codec as A

_MODEL = None
_TORCH = None
_LOAD_TRIED = False
_DEFAULT_PATHS = [
    os.environ.get("IL_POLICY_PATH", ""),
    os.path.join(os.path.dirname(__file__), "..", "models", "policy.pt"),
    os.path.join(os.path.dirname(__file__), "policy.pt"),
    "/kaggle_simulations/agent/models/policy.pt",
    "/kaggle_simulations/agent/policy.pt",
]


def _try_load():
    global _MODEL, _TORCH, _LOAD_TRIED
    _LOAD_TRIED = True
    try:
        import torch
        from .model import PolicyNet
        _TORCH = torch
        for p in _DEFAULT_PATHS:
            if p and os.path.exists(p):
                ckpt = torch.load(p, map_location="cpu")
                net = PolicyNet(c_board=ckpt.get("c_board"), g_global=ckpt.get("g_global"))
                net.load_state_dict(ckpt["state_dict"])
                net.eval()
                _MODEL = net
                return
    except Exception:  # noqa: BLE001
        _MODEL = None


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _policy_action(obs, seat, max_orders):
    torch = _TORCH
    farm = obs["farms"][seat]
    board, glob = F.encode_obs(obs, seat)
    hands = farm.get("hands", [])
    positions = [farm["farmer"]] + list(hands)
    U = len(positions)
    upos = np.zeros((1, max(U, 1), 3), np.int16)
    for i, (x, y) in enumerate(positions):
        upos[0, i] = [int(x), int(y), 1 if i == 0 else 0]
    umask = np.zeros((1, max(U, 1)), np.uint8)
    umask[0, :U] = 1
    with torch.no_grad():
        bt = torch.from_numpy(board[None]).float()
        gt = torch.from_numpy(glob[None]).float()
        pt = torch.from_numpy(upos).long()
        mt = torch.from_numpy(umask).long()
        ul, ml = _MODEL(bt, gt, pt, mt)

    def am(x):
        return int(np.argmax(x))

    unit_acts = []
    for i in range(U):
        t = am(ul["type"][0, i].numpy())
        crop = am(ul["crop"][0, i].numpy())
        item = am(ul["item"][0, i].numpy())
        cnt = am(ul["count"][0, i].numpy())
        unit_acts.append(A.decode_unit_action(t, crop, item, cnt))
    farmer_act = unit_acts[0] if unit_acts else ["PASS"]
    hand_acts = unit_acts[1:]

    pred = {
        "sell_present": _sig(ml["sell_present"][0].numpy()) > 0.5,
        "sell_qty": np.argmax(ml["sell_qty"][0].numpy(), -1),
        "bprod_present": _sig(ml["bprod_present"][0].numpy()) > 0.5,
        "bprod_qty": np.argmax(ml["bprod_qty"][0].numpy(), -1),
        "bseed_present": _sig(ml["bseed_present"][0].numpy()) > 0.5,
        "bseed_qty": np.argmax(ml["bseed_qty"][0].numpy(), -1),
        "banim_present": _sig(ml["banim_present"][0].numpy()) > 0.5,
        "banim_qty": np.argmax(ml["banim_qty"][0].numpy(), -1),
        "hire": int(np.argmax(ml["hire"][0].numpy())),
        "buy_land": bool(_sig(float(ml["buy_land"][0].numpy())) > 0.5),
    }
    market = A.decode_market(pred, max_orders=max_orders)
    return {"farmer": farmer_act, "hands": hand_acts, "market": market}


def _fallback_action(obs, seat):
    """Minimal safe heuristic: sell shed produce, otherwise PASS."""
    hands = obs["farms"][seat].get("hands", [])
    shed = obs.get("private", {}).get("shed", {})
    market = []
    for it in spec.MARKET_ITEMS:
        q = shed.get(it, 0)
        if q > 0:
            market.append(["SELL", it, int(q)])
    return {"farmer": ["PASS"], "hands": [["PASS"]] * len(hands),
            "market": market[:spec.MAX_MARKET_ORDERS]}


def act(obs, configuration=None):
    if not _LOAD_TRIED:
        _try_load()
    seat = obs.get("player", 0)
    max_orders = spec.MAX_MARKET_ORDERS
    if configuration and isinstance(configuration, dict):
        max_orders = configuration.get("maxMarketOrdersPerTurn", max_orders)
    if _MODEL is None:
        return _fallback_action(obs, seat)
    try:
        return _policy_action(obs, seat, max_orders)
    except Exception:  # noqa: BLE001
        return _fallback_action(obs, seat)

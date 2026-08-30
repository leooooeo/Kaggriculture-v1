"""Action <-> factored-label codec.

Two independent streams:
  * unit stream  : one (type, crop, item, count) label tuple per unit
  * market stream: per-step factored market labels (presence + qty buckets,
                   hire count, buy_land flag)

Decoders turn model outputs back into the dict the environment expects:
  {"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}
"""
import numpy as np

from . import spec

IGNORE = -1  # label value meaning "no supervision for this head on this sample"


# --------------------------------------------------------------------------
# Unit stream
# --------------------------------------------------------------------------
def encode_unit_action(act):
    """act: e.g. ['PLANT','WHEAT'] or ['PICKUP','FERTILIZER',3] or ['WATER'].
    Returns dict of int labels for the four unit heads."""
    if not act:
        act = ["PASS"]
    verb = act[0]
    tp = spec.UNIT_ACTION_IDX.get(verb, 0)
    crop = IGNORE
    item = IGNORE
    count = IGNORE
    if verb == "PLANT":
        c = act[1] if len(act) > 1 else "WHEAT"
        crop = spec.CROPS.index(c) if c in spec.CROPS else 0
    elif verb in ("PICKUP", "PLACE"):
        it = act[1] if len(act) > 1 else spec.UNIT_ITEMS[0]
        item = spec.UNIT_ITEMS.index(it) if it in spec.UNIT_ITEMS else 0
        n = act[2] if len(act) > 2 else 1
        count = spec.qty_to_bucket(n)
    return {"type": tp, "crop": crop, "item": item, "count": count}


def decode_unit_action(type_idx, crop_idx=0, item_idx=0, count_idx=0):
    verb = spec.UNIT_ACTIONS[int(type_idx)]
    if verb == "PLANT":
        return [verb, spec.CROPS[int(crop_idx)]]
    if verb in ("PICKUP", "PLACE"):
        return [verb, spec.UNIT_ITEMS[int(item_idx)], spec.bucket_to_qty(count_idx)]
    return [verb]


# --------------------------------------------------------------------------
# Market stream
# --------------------------------------------------------------------------
# Label vector layout (all concatenated by the model into separate heads):
SELL_PRESENT = spec.SELL_SLOTS          # binary + qty
BUY_PRODUCT_SLOTS = spec.BUY_PRODUCT_SLOTS
BUY_SEED_SLOTS = spec.BUY_SEED_SLOTS
BUY_ANIMAL_SLOTS = spec.BUY_ANIMAL_SLOTS


def empty_market_labels():
    return {
        "sell_present": np.zeros(len(SELL_PRESENT), np.float32),
        "sell_qty":     np.full(len(SELL_PRESENT), IGNORE, np.int64),
        "bprod_present": np.zeros(len(BUY_PRODUCT_SLOTS), np.float32),
        "bprod_qty":     np.full(len(BUY_PRODUCT_SLOTS), IGNORE, np.int64),
        "bseed_present": np.zeros(len(BUY_SEED_SLOTS), np.float32),
        "bseed_qty":     np.full(len(BUY_SEED_SLOTS), IGNORE, np.int64),
        "banim_present": np.zeros(len(BUY_ANIMAL_SLOTS), np.float32),
        "banim_qty":     np.full(len(BUY_ANIMAL_SLOTS), IGNORE, np.int64),
        "hire":         np.int64(0),
        "buy_land":     np.float32(0.0),
    }


def encode_market(market):
    """market: list of orders like ['SELL','WHEAT',5], ['HIRE'], ['BUY_LAND'].
    Aggregates repeated orders of the same (kind,item) by summing quantity."""
    lab = empty_market_labels()
    sums = {}  # (kind,item)->qty
    hires = 0
    for o in (market or []):
        if not o:
            continue
        kind = o[0]
        if kind == "HIRE":
            hires += 1
        elif kind == "BUY_LAND":
            lab["buy_land"] = np.float32(1.0)
        elif kind in ("SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"):
            item = o[1] if len(o) > 1 else None
            qty = o[2] if len(o) > 2 else 1
            sums[(kind, item)] = sums.get((kind, item), 0) + qty
    lab["hire"] = np.int64(min(spec.MAX_HIRE, hires))

    def _fill(kind, slots, pkey, qkey):
        for i, item in enumerate(slots):
            q = sums.get((kind, item))
            if q:
                lab[pkey][i] = 1.0
                lab[qkey][i] = spec.qty_to_bucket(q)
    _fill("SELL", SELL_PRESENT, "sell_present", "sell_qty")
    _fill("BUY_PRODUCT", BUY_PRODUCT_SLOTS, "bprod_present", "bprod_qty")
    _fill("BUY_SEED", BUY_SEED_SLOTS, "bseed_present", "bseed_qty")
    _fill("BUY_ANIMAL", BUY_ANIMAL_SLOTS, "banim_present", "banim_qty")
    return lab


def decode_market(pred, max_orders=spec.MAX_MARKET_ORDERS):
    """pred: dict with same keys as empty_market_labels but *_present are
    booleans/prob>0.5, *_qty are bucket indices, hire is an int, buy_land bool.
    Returns an ordered, capped market list."""
    orders = []

    def _emit(kind, slots, present, qty):
        for i, item in enumerate(slots):
            if present[i]:
                n = spec.bucket_to_qty(qty[i]) if qty is not None else 1
                orders.append([kind, item, int(n)])

    groups = {
        "SELL": lambda: _emit("SELL", SELL_PRESENT, pred["sell_present"], pred["sell_qty"]),
        "BUY_ANIMAL": lambda: _emit("BUY_ANIMAL", BUY_ANIMAL_SLOTS, pred["banim_present"], pred["banim_qty"]),
        "BUY_SEED": lambda: _emit("BUY_SEED", BUY_SEED_SLOTS, pred["bseed_present"], pred["bseed_qty"]),
        "BUY_PRODUCT": lambda: _emit("BUY_PRODUCT", BUY_PRODUCT_SLOTS, pred["bprod_present"], pred["bprod_qty"]),
    }
    for kind in spec.MARKET_DECODE_ORDER:
        if kind == "BUY_LAND":
            if pred.get("buy_land"):
                orders.append(["BUY_LAND"])
        elif kind == "HIRE":
            for _ in range(int(pred.get("hire", 0))):
                orders.append(["HIRE"])
        elif kind in groups:
            groups[kind]()
    return orders[:max_orders]

"""Observation -> tensor feature encoding.

Pure numpy so the data pipeline and the runtime agent's preprocessing run
without torch.  Produces:
  board : (C, H, W) float32   spatial map
  glob  : (G,)      float32   global scalar features
Unit-level features are derived at gather time from the board + unit position.
"""
import math
import numpy as np

from . import spec

# ---- board channel layout -------------------------------------------------
# Keep this list authoritative; model infers C from it.
BOARD_CHANNELS = (
    ["locked", "empty", "weed"]
    + [f"crop_{c}" for c in spec.CROPS]              # 5
    + ["watered", "unwatered", "crop_yield", "fertilized"]
    + ["coop", "pasture"]
    + [f"animal_{a}" for a in spec.ANIMALS]          # 3
    + ["fed", "cared", "unfed", "fert_avail",
       "animal_yield", "care_bonus"]
    + ["farmer_here", "hand_here"]
)
C_BOARD = len(BOARD_CHANNELS)
_CH = {name: i for i, name in enumerate(BOARD_CHANNELS)}


def _log1p_norm(x, scale):
    return math.log1p(max(0.0, float(x))) / math.log1p(scale)


def encode_board(farm, size=spec.BOARD_SIZE, day=0):
    """farm: obs['farms'][seat] dict. Returns (C,H,W) float32."""
    b = np.zeros((C_BOARD, size, size), dtype=np.float32)
    tiles = farm["tiles"]
    for y in range(size):
        row = tiles[y]
        for x in range(size):
            t = row[x]
            if t is None:
                b[_CH["empty"], y, x] = 1.0
                continue
            if t == "LOCKED":
                b[_CH["locked"], y, x] = 1.0
                continue
            kind = t.get("kind")
            if kind == "WEED":
                b[_CH["weed"], y, x] = 1.0
            elif kind == "PLANT":
                crop = t.get("crop")
                if crop in spec.CROPS:
                    b[_CH[f"crop_{crop}"], y, x] = 1.0
                b[_CH["watered"], y, x] = 1.0 if t.get("watered_today") else 0.0
                b[_CH["unwatered"], y, x] = min(2, t.get("consecutive_unwatered", 0)) / 2.0
                b[_CH["crop_yield"], y, x] = min(6, t.get("yield_units", 0)) / 6.0
                b[_CH["fertilized"], y, x] = 1.0 if t.get("fertilized_until_day", -1) >= day else 0.0
            elif kind in ("COOP", "PASTURE"):
                b[_CH["coop" if kind == "COOP" else "pasture"], y, x] = 1.0
                animal = t.get("animal")
                if animal in spec.ANIMALS:
                    b[_CH[f"animal_{animal}"], y, x] = 1.0
                b[_CH["fed"], y, x] = 1.0 if t.get("fed_today") else 0.0
                b[_CH["cared"], y, x] = 1.0 if t.get("cared_today") else 0.0
                b[_CH["unfed"], y, x] = min(2, t.get("consecutive_unfed", 0)) / 2.0
                b[_CH["fert_avail"], y, x] = 1.0 if t.get("fertilizer_available") else 0.0
                b[_CH["animal_yield"], y, x] = min(6, t.get("yield_units", 0)) / 6.0
                b[_CH["care_bonus"], y, x] = min(6, t.get("pending_care_bonus", 0)) / 6.0
    fx, fy = farm["farmer"]
    if 0 <= fx < size and 0 <= fy < size:
        b[_CH["farmer_here"], fy, fx] += 1.0
    for hx, hy in farm["hands"]:
        if 0 <= hx < size and 0 <= hy < size:
            b[_CH["hand_here"], hy, hx] += 1.0
    return b


# ---- global feature layout ------------------------------------------------
GLOBAL_NAMES = (
    ["day_frac", "hour_frac", "money_log", "opp_money_log", "n_hands",
     "hires_today", "n_quadrants"]
    + [f"q_{q}" for q in spec.QUADRANTS]                      # 4
    + [f"price_{it}" for it in spec.MARKET_ITEMS]             # 9
    + [f"inv_{it}" for it in spec.MARKET_ITEMS]               # 9
    + [f"shed_{it}" for it in spec.SHED_ITEMS]               # 12
    + [f"seed_{c}" for c in spec.CROPS]                       # 5
    + [f"shop_{s}" for s in spec.SHOP_TYPES]                  # 8
)
G_GLOBAL = len(GLOBAL_NAMES)


def encode_global(obs, seat):
    farm = obs["farms"][seat]
    opp = obs["farms"][1 - seat] if len(obs["farms"]) > 1 else farm
    g = np.zeros(G_GLOBAL, dtype=np.float32)
    idx = {n: i for i, n in enumerate(GLOBAL_NAMES)}
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    g[idx["day_frac"]] = day / spec.SEASON_DAYS
    g[idx["hour_frac"]] = hour / spec.TURNS_PER_DAY
    g[idx["money_log"]] = _log1p_norm(farm.get("money", 0), 200000)
    g[idx["opp_money_log"]] = _log1p_norm(opp.get("money", 0), 200000)
    g[idx["n_hands"]] = min(20, len(farm.get("hands", []))) / 20.0
    g[idx["hires_today"]] = min(20, farm.get("hires_today", 0)) / 20.0
    quads = set(farm.get("unlocked_quadrants", []))
    g[idx["n_quadrants"]] = len(quads) / 4.0
    for q in spec.QUADRANTS:
        g[idx[f"q_{q}"]] = 1.0 if q in quads else 0.0
    prices = obs["market"].get("prices", {})
    inv = obs["market"].get("inventory", {})
    for it in spec.MARKET_ITEMS:
        g[idx[f"price_{it}"]] = prices.get(it, spec.BASE_PRICE[it]) / (2.0 * spec.BASE_PRICE[it])
        g[idx[f"inv_{it}"]] = inv.get(it, spec.I0) / (2.0 * spec.I0)
    shed = obs.get("private", {}).get("shed", {})
    for it in spec.SHED_ITEMS:
        g[idx[f"shed_{it}"]] = min(100, shed.get(it, 0)) / 100.0
    seeds = obs.get("private", {}).get("seeds", {})
    for c in spec.CROPS:
        g[idx[f"seed_{c}"]] = min(30, seeds.get(c, 0)) / 30.0
    shops = obs.get("town", {}).get("unlocked_shops", [])
    for s in shops:
        key = f"shop_{s}"
        if key in idx:
            g[idx[key]] = min(4, g[idx[key]] * 4 + 1) / 4.0
    return g


def encode_obs(obs, seat):
    """Returns (board, glob)."""
    size = len(obs["farms"][seat]["tiles"])
    board = encode_board(obs["farms"][seat], size=size, day=obs.get("day", 0))
    glob = encode_global(obs, seat)
    return board, glob

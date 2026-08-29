"""Turn replay JSON files into training shards (one .npz per replay).

Usage:
    python -m il.dataset --replays data/replays --out data/shards \
        --team "TARGET_TEAM_NAME"
If --team is omitted, the highest-scoring seat of each replay is used (handy
for "imitate whoever won"), but for a focused clone pass the exact team name.
"""
import argparse
import glob
import json
import os

import numpy as np

from . import spec
from . import features as F
from . import action_codec as A

U_MAX = 32  # padded units per step (farmer + up to 31 hands)


def _seat_for_team(replay, team):
    names = replay["info"]["TeamNames"]
    if team is None:
        rw = replay.get("rewards") or [0, 0]
        rw = [(-1 if r is None else r) for r in rw]
        return int(np.argmax(rw))
    if team in names:
        return names.index(team)
    return None


def _valid_obs(obs, seat):
    return (isinstance(obs, dict) and "farms" in obs and "market" in obs
            and "private" in obs and seat < len(obs["farms"])
            and isinstance(obs["farms"][seat], dict)
            and "tiles" in obs["farms"][seat])


def encode_replay(path, team):
    replay = json.load(open(path))
    seat = _seat_for_team(replay, team)
    if seat is None:
        return None
    boards, globs = [], []
    upos = []   # (U,3) x,y,is_farmer
    ulab = []   # (U,4) type,crop,item,count
    umask = []
    mk = {k: [] for k in A.empty_market_labels()}
    steps = replay["steps"]
    # In kaggle-environments, steps[t]['action'] is the action applied to
    # steps[t-1]'s observation to produce steps[t]. So the correct behavioral-
    # cloning pair is (obs at t-1) -> (action at t).
    for t in range(1, len(steps)):
        obs = steps[t - 1][seat].get("observation")
        act = steps[t][seat].get("action")
        if not _valid_obs(obs, seat) or not act:
            continue
        farm = obs["farms"][seat]
        board, glob = F.encode_obs(obs, seat)

        # ---- units ----
        units = [(farm["farmer"], act.get("farmer", ["PASS"]), 1)]
        hands = farm.get("hands", [])
        hacts = act.get("hands", [])
        for i, hpos in enumerate(hands):
            ha = hacts[i] if i < len(hacts) else ["PASS"]
            units.append((hpos, ha, 0))
        p = np.zeros((U_MAX, 3), np.int16)
        l = np.full((U_MAX, 4), A.IGNORE, np.int16)
        m = np.zeros((U_MAX,), np.uint8)
        for i, (pos, ua, is_farmer) in enumerate(units[:U_MAX]):
            lab = A.encode_unit_action(ua)
            p[i] = [int(pos[0]), int(pos[1]), is_farmer]
            l[i] = [lab["type"], lab["crop"], lab["item"], lab["count"]]
            m[i] = 1

        # ---- market ----
        ml = A.encode_market(act.get("market", []))

        boards.append(board); globs.append(glob)
        upos.append(p); ulab.append(l); umask.append(m)
        for k in mk:
            mk[k].append(ml[k])

    if not boards:
        return None
    out = {
        "board": np.asarray(boards, np.float32),
        "glob": np.asarray(globs, np.float32),
        "upos": np.asarray(upos, np.int16),
        "ulab": np.asarray(ulab, np.int16),
        "umask": np.asarray(umask, np.uint8),
    }
    for k in mk:
        out[f"mk_{k}"] = np.asarray(mk[k])
    out["_meta"] = np.asarray([seat, len(boards)], np.int64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="data/replays")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--team", default=None,
                    help="Exact TeamName to imitate. Omit = winner of each game.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.replays, "*.json")))
    if not paths:
        raise SystemExit(f"No replays found in {args.replays}")
    n_steps = 0
    kept = 0
    for pth in paths:
        try:
            enc = encode_replay(pth, args.team)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {os.path.basename(pth)}: {e}")
            continue
        if enc is None:
            continue
        name = os.path.splitext(os.path.basename(pth))[0]
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **enc)
        n = int(enc["_meta"][1])
        n_steps += n
        kept += 1
        print(f"  + {name}: seat={int(enc['_meta'][0])} steps={n}")
    print(f"Done. {kept} shards, {n_steps} step-samples -> {args.out}")


if __name__ == "__main__":
    main()

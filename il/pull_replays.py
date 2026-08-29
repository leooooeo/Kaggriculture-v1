"""Pull the leaderboard #1 team's episode replays into data/replays/.

Uses Kaggle's internal EpisodeService (the same unauthenticated /api/i/
endpoints your original snippet used), because simulation replays are public.
Unlike agent *logs*, GetEpisodeReplay returns the full replay WITH `steps`
(per-turn observation + action) — which is exactly what il/dataset.py consumes.

Usage (Colab):
    python -m il.pull_replays --creds "/content/kaggle (1).json" \
        --out data/replays --n 60
    # target a specific team instead of auto rank-1:
    python -m il.pull_replays --creds kaggle.json --team "Some Team" --n 60
    # if leaderboard team-id lookup fails, seed from a known episode of theirs:
    python -m il.pull_replays --creds kaggle.json --seed-episode 93600012 \
        --team "One-For-All" --n 30

Writes files named episode-<id>-replay.json (dataset.py globs *.json).
"""
import argparse
import json
import os
import time

import requests

COMP = "kaggriculture"
BASE = "https://www.kaggle.com/api/i/competitions.EpisodeService/"


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0",
                      "Content-Type": "application/json"})
    return s


def _post(s, method, payload):
    r = s.post(BASE + method, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()


def _authenticate(creds_path):
    with open(creds_path) as f:
        creds = json.load(f)
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _entry_fields(e):
    """Pull (teamId, teamName) from a leaderboard entry of unknown shape."""
    d = e if isinstance(e, dict) else None
    if d is None:
        for m in ("to_dict", "_asdict"):
            if hasattr(e, m):
                try:
                    d = getattr(e, m)()
                    break
                except Exception:  # noqa: BLE001
                    pass
    if d is None:
        d = getattr(e, "__dict__", {}) or {}

    def pick(*names):
        for n in names:
            if isinstance(d, dict) and d.get(n) not in (None, ""):
                return d[n]
            v = getattr(e, n, None)
            if v not in (None, ""):
                return v
        return None
    tid = pick("teamId", "team_id", "teamNameId", "id")
    name = pick("teamName", "team_name", "displayName", "name")
    return tid, name


def _iter_entries(lb):
    """Yield entries from a list, or a paginated object's submissions/teams."""
    if isinstance(lb, (list, tuple)):
        yield from lb
        return
    for attr in ("submissions", "teams", "entries", "results", "leaderboard"):
        v = getattr(lb, attr, None)
        if isinstance(v, (list, tuple)):
            yield from v
            return
    if isinstance(lb, dict):
        for attr in ("submissions", "teams", "entries", "results"):
            if isinstance(lb.get(attr), (list, tuple)):
                yield from lb[attr]
                return
    try:  # last resort: maybe it is directly iterable
        yield from lb
    except TypeError:
        return


def _leaderboard_rows_via_csv(api):
    """Most stable path: download the leaderboard CSV and parse (id,name)."""
    import csv
    import tempfile
    import zipfile
    rows = []
    with tempfile.TemporaryDirectory() as td:
        try:
            api.competition_leaderboard_download(COMP, td)
        except Exception as ex:  # noqa: BLE001
            print(f"  (leaderboard CSV download failed: {ex})")
            return rows
        files = []
        for f in os.listdir(td):
            fp = os.path.join(td, f)
            if f.endswith(".zip"):
                with zipfile.ZipFile(fp) as z:
                    z.extractall(td)
        for f in os.listdir(td):
            if f.endswith(".csv"):
                files.append(os.path.join(td, f))
        for fp in files:
            with open(fp, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                cols = {c.lower(): c for c in (reader.fieldnames or [])}
                id_col = next((cols[c] for c in cols if "team" in c and "id" in c), None)
                name_col = next((cols[c] for c in cols
                                 if "team" in c and ("name" in c or c == "team")), None)
                for r in reader:
                    tid = r.get(id_col) if id_col else None
                    name = r.get(name_col) if name_col else None
                    if tid and name:
                        rows.append((tid, name))
    return rows


def _top_team(api, rank_team):
    """Return (team_id, team_name) for the requested team, or rank #1."""
    rows = _leaderboard_rows_via_csv(api)
    if not rows:
        lb = api.competition_leaderboard_view(COMP)
        for e in _iter_entries(lb):
            tid, name = _entry_fields(e)
            if tid is not None and name is not None:
                rows.append((tid, name))
    if not rows:
        lb = api.competition_leaderboard_view(COMP)
        print(f"! Could not parse leaderboard object (type={type(lb).__name__}).")
        sample = next(iter(_iter_entries(lb)), None)
        if sample is not None:
            print("  first entry type:", type(sample).__name__,
                  "| attrs:", [a for a in dir(sample) if not a.startswith('_')][:20])
        raise SystemExit("Leaderboard parse failed; pass --team and --seed-episode.")
    if rank_team:
        for tid, name in rows:
            if name == rank_team:
                return tid, name
        print(f"! '{rank_team}' not found on leaderboard; using rank #1 ({rows[0][1]!r}).")
    return rows[0]


def _list_episode_ids(s, team_id=None, submission_id=None, seed_episode=None,
                      team_name=None, want=60):
    """Collect episode ids for a team, newest first."""
    # Resolve a submissionId to page through, either directly, from teamId, or
    # from a seed episode (the proven publicLeaderboardSubmissionId path).
    if submission_id is None and seed_episode is not None:
        meta = _post(s, "ListEpisodes", {"ids": [int(seed_episode)]})
        teams = meta.get("teams", [])
        tid = team_id
        if tid is None and team_name:
            tid = next((t["id"] for t in teams if t.get("teamName") == team_name), None)
        if tid is not None:
            for ep in meta.get("episodes", []):
                for a in ep.get("agents", []):
                    if a.get("teamId") == tid and a.get("submissionId"):
                        submission_id = a["submissionId"]
                        break
                if submission_id:
                    break
            if submission_id is None:
                submission_id = next(
                    (t.get("publicLeaderboardSubmissionId") for t in teams
                     if t.get("id") == tid), None)

    def _page(payload):
        data = _post(s, "ListEpisodes", payload)
        return data.get("episodes", [])

    seen, eps = set(), []
    # Prefer teamId listing; fall back to submissionId.
    payloads = []
    if team_id is not None:
        payloads.append({"teamId": int(team_id)})
    if submission_id is not None:
        payloads.append({"submissionId": int(submission_id)})
    if not payloads:
        raise SystemExit("No teamId/submissionId resolved; pass --seed-episode.")

    for base_payload in payloads:
        payload = dict(base_payload)
        try:
            while True:
                batch = _page(payload)
                new = [e for e in batch if e["id"] not in seen]
                for e in new:
                    seen.add(e["id"])
                eps.extend(new)
                if not new or len(eps) >= want * 4:
                    break
                payload["before"] = min(e["id"] for e in batch)
        except requests.HTTPError as ex:
            print(f"  (listing via {list(base_payload)[0]} failed: {ex})")
            continue
        if eps:
            break
    return sorted({e["id"] for e in eps}, reverse=True)


def _get_replay(s, episode_id):
    data = _post(s, "GetEpisodeReplay", {"episodeId": int(episode_id)})
    rep = data.get("replay", data)
    if isinstance(rep, str):
        rep = json.loads(rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creds", required=True, help="path to kaggle.json")
    ap.add_argument("--out", default="data/replays")
    ap.add_argument("--team", default=None,
                    help="Exact team name; omit to use current leaderboard #1.")
    ap.add_argument("--n", type=int, default=60, help="how many recent episodes")
    ap.add_argument("--seed-episode", type=int, default=None,
                    help="fallback: a known episode id involving the team")
    ap.add_argument("--sleep", type=float, default=0.8)
    args = ap.parse_args()

    api = _authenticate(args.creds)
    s = _session()

    team_id, team_name = None, args.team
    try:
        team_id, team_name = _top_team(api, args.team)
        print(f"Target team: {team_name!r} (teamId={team_id})")
    except SystemExit as e:
        if args.seed_episode is None:
            raise
        print(f"! leaderboard lookup failed ({e}); using seed episode path.")

    ids = _list_episode_ids(s, team_id=team_id, seed_episode=args.seed_episode,
                            team_name=team_name, want=args.n)
    ids = ids[:args.n]
    print(f"Found {len(ids)} episodes; downloading replays...")
    os.makedirs(args.out, exist_ok=True)
    json.dump(ids, open(os.path.join(args.out, "_episode_ids.json"), "w"))

    ok = 0
    for i, ep in enumerate(ids):
        dst = os.path.join(args.out, f"episode-{ep}-replay.json")
        if os.path.exists(dst):
            ok += 1
            continue
        try:
            rep = _get_replay(s, ep)
            if "steps" not in rep or "info" not in rep:
                print(f"  ! ep {ep}: replay missing steps/info, skipped")
                continue
            json.dump(rep, open(dst, "w"))
            ok += 1
            print(f"[{i+1}/{len(ids)}] ep {ep} saved ({len(rep['steps'])} steps)")
        except Exception as e:  # noqa: BLE001
            print(f"  ! ep {ep}: {e!r}")
        time.sleep(args.sleep)
    print(f"Done. {ok}/{len(ids)} replays in {args.out}")


if __name__ == "__main__":
    main()

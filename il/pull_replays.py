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


def _session(creds=None):
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0",
                      "Content-Type": "application/json"})
    # Basic auth (username, key) — needed for team-scoped internal endpoints;
    # by-id replay fetches work without it, listing a team's history may not.
    if creds and creds.get("username") and creds.get("key"):
        s.auth = (creds["username"], creds["key"])
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
    return api, creds


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
    sub = pick("submissionId", "submission_id", "bestSubmissionId",
               "publicLeaderboardSubmissionId", "lastSubmissionId")
    return tid, name, sub


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
                sub_col = next((cols[c] for c in cols
                                if "submission" in c and "id" in c), None)
                for r in reader:
                    tid = r.get(id_col) if id_col else None
                    name = r.get(name_col) if name_col else None
                    sub = r.get(sub_col) if sub_col else None
                    if tid and name:
                        rows.append((tid, name, sub))
    return rows


def _top_team(api, rank_team):
    """Return (team_id, team_name) for the requested team, or rank #1."""
    rows = _leaderboard_rows_via_csv(api)
    if not rows:
        lb = api.competition_leaderboard_view(COMP)
        for e in _iter_entries(lb):
            tid, name, sub = _entry_fields(e)
            if tid is not None and name is not None:
                rows.append((tid, name, sub))
    if not rows:
        lb = api.competition_leaderboard_view(COMP)
        print(f"! Could not parse leaderboard object (type={type(lb).__name__}).")
        sample = next(iter(_iter_entries(lb)), None)
        if sample is not None:
            print("  first entry type:", type(sample).__name__,
                  "| attrs:", [a for a in dir(sample) if not a.startswith('_')][:20])
        raise SystemExit("Leaderboard parse failed; pass --team and --seed-episode.")
    chosen = None
    if rank_team:
        for row in rows:
            if row[1] == rank_team:
                chosen = row
                break
        if chosen is None:
            print(f"! '{rank_team}' not found on leaderboard; using rank #1 ({rows[0][1]!r}).")
    chosen = chosen or rows[0]
    tid, name, sub = chosen
    # If the (CSV) row had no submissionId, try the view object which sometimes
    # carries it, matching by team name.
    if not sub:
        try:
            lb = api.competition_leaderboard_view(COMP)
            for e in _iter_entries(lb):
                etid, ename, esub = _entry_fields(e)
                if esub and (ename == name or str(etid) == str(tid)):
                    sub = esub
                    break
        except Exception:  # noqa: BLE001
            pass
    return tid, name, sub


def _list_episode_ids(s, team_id=None, submission_id=None, seed_episode=None,
                      team_name=None, want=60):
    """Collect episode ids for a team, newest first."""
    # Resolve a submissionId to page through, either directly, from teamId, or
    # from a seed episode (the proven publicLeaderboardSubmissionId path).
    if submission_id is None and seed_episode is not None:
        meta = _post(s, "ListEpisodes", {"ids": [int(seed_episode)]})
        teams = meta.get("teams", [])
        episodes = meta.get("episodes", [])
        print(f"  [seed {seed_episode}] top keys: {list(meta.keys())}")
        if teams:
            print(f"  [seed] team keys: {list(teams[0].keys())}")
            print(f"  [seed] teams: " +
                  "; ".join(f"{t.get('teamName')}#{t.get('id')}"
                            f"->sub{t.get('publicLeaderboardSubmissionId') or t.get('submissionId')}"
                            for t in teams))
        if episodes and episodes[0].get("agents"):
            print(f"  [seed] agent keys: {list(episodes[0]['agents'][0].keys())}")

        def _team_matches(t):
            return (str(t.get("id")) == str(team_id)
                    or (team_name and t.get("teamName") == team_name))

        # 1) submissionId straight off the matching team record
        for t in teams:
            if _team_matches(t):
                submission_id = (t.get("publicLeaderboardSubmissionId")
                                 or t.get("submissionId")
                                 or t.get("lastSubmissionId"))
                if submission_id:
                    break
        # 2) else from the agents of the seed episode(s)
        if submission_id is None:
            tid = team_id or next((t.get("id") for t in teams if _team_matches(t)), None)
            for ep in episodes:
                for a in ep.get("agents", []):
                    a_tid = a.get("teamId") or a.get("teamNameId")
                    if a.get("submissionId") and (a_tid == tid or str(a_tid) == str(tid)):
                        submission_id = a["submissionId"]
                        break
                if submission_id:
                    break
        # 3) last resort: if exactly one team isn't ours, the other is ours
        if submission_id is None and len(teams) == 2 and team_name:
            other = [t for t in teams if t.get("teamName") == team_name]
            if other:
                submission_id = (other[0].get("publicLeaderboardSubmissionId")
                                 or other[0].get("submissionId"))
        if submission_id:
            print(f"  [seed] resolved submissionId = {submission_id}")

    def _page(payload):
        data = _post(s, "ListEpisodes", payload)
        return data.get("episodes", [])

    seen, eps = set(), []
    # ListEpisodes accepts submissionId (teamId returns 400), so prefer it.
    payloads = []
    if submission_id is not None:
        payloads.append({"submissionId": int(submission_id)})
    if team_id is not None:
        payloads.append({"teamId": int(team_id)})  # last resort; may 400
    if not payloads:
        raise SystemExit("No submissionId resolved; pass --team and --seed-episode.")

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


def _download_replay(api, episode_id, out_dir):
    """Use the official kaggle method (handles the GCS-hosted replay), then
    normalize the produced file to episode-<id>-replay.json and return it."""
    before = set(os.listdir(out_dir))
    api.competition_episode_replay(int(episode_id), path=out_dir, quiet=True)
    after = set(os.listdir(out_dir))
    cand = [f for f in (after - before) if f.endswith(".json")]
    if not cand:
        cand = [f for f in after
                if str(episode_id) in f and f.endswith(".json")]
    for f in cand:
        fp = os.path.join(out_dir, f)
        try:
            rep = json.load(open(fp))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(rep, dict) and "steps" in rep:
            dst = os.path.join(out_dir, f"episode-{episode_id}-replay.json")
            if os.path.abspath(fp) != os.path.abspath(dst):
                os.replace(fp, dst)
            return rep
    return None


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

    api, creds = _authenticate(args.creds)
    s = _session(creds)

    team_id, team_name, submission_id = None, args.team, None
    try:
        team_id, team_name, submission_id = _top_team(api, args.team)
        print(f"Target team: {team_name!r} (teamId={team_id}, submissionId={submission_id})")
    except SystemExit as e:
        if args.seed_episode is None:
            raise
        print(f"! leaderboard lookup failed ({e}); using seed episode path.")

    ids = _list_episode_ids(s, team_id=team_id, submission_id=submission_id,
                            seed_episode=args.seed_episode,
                            team_name=team_name, want=args.n)
    ids = ids[:args.n]
    if not ids:
        raise SystemExit(
            "\nNo episodes listed for this team.\n"
            f"The internal API needs a submissionId (teamId listing is blocked) and\n"
            f"none was found automatically for {team_name!r}.\n"
            "Fix: open this team on the Kaggle leaderboard, click any of their games,\n"
            "copy the episode id from the URL, and re-run with the seed fallback:\n"
            f'  python -m il.pull_replays --creds "{args.creds}" '
            f'--team "{team_name}" --seed-episode <EPISODE_ID> --n {args.n}\n')
    print(f"Found {len(ids)} episodes; downloading replays...")
    os.makedirs(args.out, exist_ok=True)
    # write the id list OUTSIDE the replays dir so dataset.py won't glob it
    ids_path = os.path.join(os.path.dirname(args.out.rstrip("/")) or ".",
                            "_episode_ids.json")
    json.dump(ids, open(ids_path, "w"))

    ok = 0
    for i, ep in enumerate(ids):
        dst = os.path.join(args.out, f"episode-{ep}-replay.json")
        if os.path.exists(dst):
            ok += 1
            continue
        try:
            rep = _download_replay(api, ep, args.out)
            if rep is None:
                print(f"  ! ep {ep}: no replay json produced")
                continue
            if ok == 0:  # confirm structure once
                info = rep.get("info", {})
                print(f"  [check] keys={list(rep.keys())} | "
                      f"info.TeamNames={info.get('TeamNames')}")
            ok += 1
            print(f"[{i+1}/{len(ids)}] ep {ep} saved ({len(rep['steps'])} steps)")
        except Exception as e:  # noqa: BLE001
            print(f"  ! ep {ep}: {e!r}")
        time.sleep(args.sleep)
    print(f"Done. {ok}/{len(ids)} replays in {args.out}")


if __name__ == "__main__":
    main()

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


def _top_team(api, rank_team):
    """Return (team_id, team_name) for the requested team, or rank #1."""
    lb = api.competition_leaderboard_view(COMP)
    rows = []
    for e in lb:
        d = e if isinstance(e, dict) else getattr(e, "__dict__", {})
        tid = d.get("teamId") or d.get("teamNameId") or getattr(e, "teamId", None)
        name = d.get("teamName") or getattr(e, "teamName", None)
        if tid is not None and name is not None:
            rows.append((tid, name))
    if not rows:
        raise SystemExit("Could not parse leaderboard; pass --team and --seed-episode.")
    if rank_team:
        for tid, name in rows:
            if name == rank_team:
                return tid, name
        print(f"! '{rank_team}' not found on leaderboard; using rank #1.")
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

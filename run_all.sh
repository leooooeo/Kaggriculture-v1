#!/usr/bin/env bash
# One-shot pipeline: pull #1 replays -> build shards -> train -> evaluate -> package.
# Run after cloning the repo. Requires a Kaggle token json.
#
#   bash run_all.sh /content/kaggle.json          # auto-target leaderboard #1
#   TEAM="Some Team" bash run_all.sh kaggle.json  # target a specific team
#   N=80 EPOCHS=80 bash run_all.sh kaggle.json    # override sizes
set -euo pipefail

CREDS="${1:?usage: bash run_all.sh <path/to/kaggle.json>}"
TEAM="${TEAM:-}"                 # empty = leaderboard rank #1
N="${N:-60}"                     # episodes to pull
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-512}"
OUT_MODEL="${OUT_MODEL:-models/policy.pt}"

echo "==> deps"
pip install -q numpy torch kaggle-environments kaggle

TEAM_ARG=(); [ -n "$TEAM" ] && TEAM_ARG=(--team "$TEAM")

echo "==> 1/4 pull replays (team='${TEAM:-#1}', n=$N)"
python -m il.pull_replays --creds "$CREDS" --out data/replays --n "$N" "${TEAM_ARG[@]}"

# Resolve the team name that was actually pulled so dataset clones that seat.
CLONE_TEAM="$TEAM"
if [ -z "$CLONE_TEAM" ]; then
  CLONE_TEAM="$(python - <<'PY'
import glob, json
# most common team across pulled replays that is NOT always the opponent:
# pick the team present in the most games (the target plays them all).
from collections import Counter
c = Counter()
for f in glob.glob("data/replays/episode-*-replay.json"):
    for n in json.load(open(f))["info"]["TeamNames"]:
        c[n] += 1
print(c.most_common(1)[0][0] if c else "")
PY
)"
fi
echo "    cloning seat of team: '$CLONE_TEAM'"

echo "==> 2/4 build shards"
python -m il.dataset --replays data/replays --out data/shards --team "$CLONE_TEAM"

echo "==> 3/4 train ($EPOCHS epochs)"
python -m il.train --shards data/shards --out "$OUT_MODEL" --epochs "$EPOCHS" --batch "$BATCH"

echo "==> 4/4 evaluate + package"
python -m il.evaluate --games 6 --opponent random || true
cp "$OUT_MODEL" models/policy.pt 2>/dev/null || true
tar -czf submission.tar.gz main.py il/ models/policy.pt
echo "Built submission.tar.gz — submit with:"
echo "  kaggle competitions submit kaggriculture -f submission.tar.gz -m 'IL clone v1'"

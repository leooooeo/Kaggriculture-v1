"""Evaluate the trained policy in the real environment against baselines.

    python -m il.evaluate --games 6 --opponent random
    python -m il.evaluate --games 4 --opponent new.py
"""
import argparse

from kaggle_environments import make

from . import policy_agent as PA


def _me(o, c=None):
    return PA.act(o, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--opponent", default="random")
    ap.add_argument("--steps", type=int, default=720)
    args = ap.parse_args()

    wins = ties = 0
    my_scores, opp_scores = [], []
    for g in range(args.games):
        env = make("kaggriculture", configuration={"episodeSteps": args.steps})
        # alternate seats to remove first-player bias
        agents = [_me, args.opponent] if g % 2 == 0 else [args.opponent, _me]
        env.run(agents)
        fin = env.steps[-1]
        r = [s.get("reward") or 0 for s in fin]
        mine = r[0] if g % 2 == 0 else r[1]
        opp = r[1] if g % 2 == 0 else r[0]
        my_scores.append(mine); opp_scores.append(opp)
        if mine > opp:
            wins += 1
        elif mine == opp:
            ties += 1
        print(f"game {g+1}: me={mine:.0f} opp={opp:.0f} {'WIN' if mine>opp else ('TIE' if mine==opp else 'loss')}")
    n = args.games
    print(f"\nvs {args.opponent}: {wins}/{n} wins, {ties} ties | "
          f"avg me={sum(my_scores)/n:.0f} opp={sum(opp_scores)/n:.0f}")


if __name__ == "__main__":
    main()

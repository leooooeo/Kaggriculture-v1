"""Kaggriculture submission entrypoint.

Bundle for submission:  tar -czf submission.tar.gz main.py il/ models/policy.pt
The environment calls `agent(observation, configuration)` each turn.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from il.policy_agent import act  # noqa: E402


def agent(obs, config=None):
    return act(obs, config)

"""
Streaming utilities for online learning.
"""
from .online_learner import EventBuffer, StreamProcessor, OnlineLearner

__all__ = [
    'EventBuffer',
    'StreamProcessor',
    'OnlineLearner',
]

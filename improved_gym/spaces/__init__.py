"""This module implements various spaces.

Spaces describe mathematical sets and are used in Gym to specify valid actions and observations.
Every Gym environment must have the attributes ``action_space`` and ``observation_space``.
If, for instance, three possible actions (0,1,2) can be performed in your environment and observations
are vectors in the two-dimensional unit cube, the environment code may contain the following two lines::

    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(0, 1, shape=(2,))
"""
from improved_gym.spaces.box import Box
from improved_gym.spaces.dict import Dict
from improved_gym.spaces.discrete import Discrete
from improved_gym.spaces.graph import Graph, GraphInstance
from improved_gym.spaces.multi_binary import MultiBinary
from improved_gym.spaces.multi_discrete import MultiDiscrete
from improved_gym.spaces.sequence import Sequence
from improved_gym.spaces.space import Space
from improved_gym.spaces.text import Text
from improved_gym.spaces.tuple import Tuple
from improved_gym.spaces.utils import flatdim, flatten, flatten_space, unflatten

__all__ = [
    "Space",
    "Box",
    "Discrete",
    "Text",
    "Graph",
    "GraphInstance",
    "MultiDiscrete",
    "MultiBinary",
    "Tuple",
    "Sequence",
    "Dict",
    "flatdim",
    "flatten_space",
    "flatten",
    "unflatten",
]

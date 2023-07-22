"""Root __init__ of the improved_gym module setting the __all__ of improved_gym modules."""
# isort: skip_file

from improved_gym import error
from improved_gym.version import VERSION as __version__

from improved_gym.core import (
    Env,
    Wrapper,
    ObservationWrapper,
    ActionWrapper,
    RewardWrapper,
)
from improved_gym.spaces import Space
from improved_gym.envs import make, spec, register
from improved_gym import logger
from improved_gym import vector
from improved_gym import wrappers
import os
import sys

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "register"]

# Initializing pygame initializes audio connections through SDL. SDL uses alsa by default on all Linux systems
# SDL connecting to alsa frequently create these giant lists of warnings every time you import an environment using
#   pygame
# DSP is far more benign (and should probably be the default in SDL anyways)

if sys.platform.startswith("linux"):
    os.environ["SDL_AUDIODRIVER"] = "dsp"

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

try:
    import gym_notices.notices as notices

    # print version warning if necessary
    notice = notices.notices.get(__version__)
    if notice:
        print(notice, file=sys.stderr)

except Exception:  # nosec
    pass

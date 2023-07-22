"""Wrapper for clipping actions within a valid bound."""
import numpy as np

import improved_gym
from improved_gym import ActionWrapper
from improved_gym.spaces import Box


class ClipAction(ActionWrapper):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import improved_gym
        >>> env = improved_gym.make('Bipedal-Walker-v3')
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env.step(np.array([5.0, 2.0, -10.0, 0.0]))
        # Executes the action np.array([1.0, 1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: improved_gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)
        super().__init__(env)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(action, self.action_space.low, self.action_space.high)

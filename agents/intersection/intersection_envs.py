# intersection_envs.py
import gymnasium as gym
import highway_env  # Needed for env registration
from highway_env.envs.intersection_env import IntersectionEnv


def make_intersection_env(
    obs_type: str = "LidarObservation",
    render_mode: str | None = None,
):
    """
    Factory that returns a configured intersection-v1 environment.

    obs_type: "LidarObservation" or "GrayscaleObservation"
    """

    def _init():
        # Start from the default config for IntersectionEnv
        config = IntersectionEnv.default_config()

        if obs_type == "LidarObservation":
            config["observation"] = {
                "type": "LidarObservation",
                "cells": 64,
                "maximum_range": 50,
                "normalize": True,
            }

        elif obs_type == "GrayscaleObservation":
            config["observation"] = {
                "type": "GrayscaleObservation",
                "weights": [0.2989, 0.5870, 0.1140],
                "observation_shape": (84, 84),
                "stack_size": 4,
            }
            config["offscreen_rendering"] = True
            config["screen_width"] = 84
            config["screen_height"] = 84

        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        # Create the env with the final config so observation_space matches from the start
        env = gym.make("intersection-v1", render_mode=render_mode, config=config)
        return env

    return _init

#!/usr/bin/env python

# requires adding a module dependency from the experiment module to multiagent-particle-envs:
# https://github.com/openai/multiagent-particle-envs.git
from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
from sarsa import SarsaLambda
import numpy as np
import os,sys

if __name__ == '__main__':
    # hyper parameters
    num_episodes = 100
    gamma = 1.
    lam = 0.5
    alpha = 0.01
    adam_lr = 0.01  # learning rate for Adam optimizer

    # load the simple_tag scenario
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    # create multi-agent environment
    # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None)

    SarsaLambda(env, gamma, lam, alpha, num_episodes)

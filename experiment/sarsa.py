#!/usr/bin/env python

# requires adding a module dependency from the experiment module to multiagent-particle-envs:
# https://github.com/openai/multiagent-particle-envs.git
from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy
import numpy as np

class RandomPolicy(Policy):
    def __init__(self, env:MultiAgentEnv, agent_index):
        self.n_action_u = env.action_space[agent_index].n
        self.n_action_c = env.world.dim_c

    def action(self, obs):
        u = np.zeros(self.n_action_u)
        c = np.zeros(self.n_action_c)
        u[np.random.randint(0, self.n_action_u)] = 1
        return np.concatenate([u, c])

def SarsaLambda(
        env: MultiAgentEnv,
        gamma: float,  # discount factor
        lam: float,  # decay rate
        alpha: float,  # step size,
        num_episode: int,
        max_episode_len=100
):
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward

    policies = [RandomPolicy(env, agent_index) for agent_index in range(env.n)]
    obs_n = env.reset()

    episode_step = 0
    while True:
        # get action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))

        # step
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        episode_step += 1
        # render
        env.render()
        # if len(episode_rewards) %100 == 0:
        #     env.render()

        for i, r in enumerate(reward_n):
            episode_rewards[-1] += r
            agent_rewards[i][-1] += r

        # display rewards
        for agent in env.world.agents:
            print(agent.name + " reward: %0.3f" % env._get_reward(agent))

        if all(done_n) or episode_step >= max_episode_len:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for agent_reward in agent_rewards:
                agent_reward.append(0)

        if len(episode_rewards) > num_episode:
            break

    ## TODO: print/save rewards summary


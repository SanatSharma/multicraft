#!/usr/bin/env python

# requires adding a module dependency from the experiment module to multiagent-particle-envs:
# https://github.com/openai/multiagent-particle-envs.git
from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy
import numpy as np


class AgentTrainer(object):
    def __init__(self, agent_name, observation_shape, action_space):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def update(self, obs, act, reward, done, terminal):
        raise NotImplemented()


class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def action(self, obs):
        u = np.zeros(self.action_space.n)
        u[np.random.randint(0, self.action_space.n)] = 1
        return u

class TileCoding:
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        pass


class FixedRandomPolicyAgentTrainer(AgentTrainer):

    def __init__(self, agent_name, observation_shape, action_space):
        self.agent_name = agent_name
        self.action_space = action_space
        self.policy = RandomPolicy(action_space)

    def action(self, obs):
        return self.policy.action(obs)

    def update(self, obs, act, reward, done, terminal):
        pass


class SarsaLambdaAgentTrainer(AgentTrainer):
    class EpsilonGreedy:
        def __init__(self, epsilon):
            self.epsilon = epsilon

        def _call_(self, obs, done, w):
            pass

    def __init__(self, agent_name, observation_shape, action_space, gamma, lam, alpha):
        self.agent_name = agent_name
        self.observation_shape = observation_shape
        self.action_space = action_space
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha

        self.policy = RandomPolicy(action_space)


    def action(self, obs):
        return self.policy.action(obs)

    def update(self, obs, act, reward, done, terminal):
        pass

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

    trainers = []
    for i, agent in enumerate(env.agents):
        if agent.adversary:
            trainers.append(SarsaLambdaAgentTrainer("agent_%d" % i, env.observation_space[i], env.action_space[i],
                                                    gamma, lam, alpha))
        else:
            trainers.append(FixedRandomPolicyAgentTrainer("agent_%d" % i, env.observation_space[i], env.action_space[i]))

    obs_n = env.reset()
    episode_step = 0
    while True:
        # get action from each agent's policy
        act_n = []
        for i, trainer in enumerate(trainers):
            act_n.append(trainer.action(obs_n[i]))

        # step
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        episode_step += 1
        terminal = episode_step >= max_episode_len

        for i, trainer in enumerate(trainers):
            trainer.update(obs_n[i], act_n[i], reward_n[i], done_n[i], terminal)

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

        if all(done_n) or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for agent_reward in agent_rewards:
                agent_reward.append(0)

        if len(episode_rewards) > num_episode:
            break

    ## TODO: print/save rewards summary


#!/usr/bin/env python

# requires adding a module dependency from the experiment module to multiagent-particle-envs:
# https://github.com/openai/multiagent-particle-envs.git
from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy
import numpy as np
import time
import os
import csv

class AgentTrainer(object):
    def __init__(self, agent_name, observation_shape, action_space):
        raise NotImplemented()

    def action(self, obs):
        raise NotImplemented()

    def update(self, obs, action, reward, next_obs, done):
        raise NotImplemented()
    
    def reset(self, obs):
        raise NotImplemented()


class InboundRandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def action(self, obs):
        u = np.zeros(self.action_space.n)
        u[np.random.randint(0, self.action_space.n)] = 1

        # makes sure an agent stays in-bound
        x, y = obs[2], obs[3]
        if x <= -1:
            u[1] = 2
        elif x >= 1:
            u[2] = 2

        if y <= -1:
            u[3] = 2
        elif y >= 1:
            u[4] = 2

        return u

class TileCoding:
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):

        self.num_tilings = num_tilings
        self.num_state_dimension = len(state_low)
        self.num_actions = num_actions
        self.tilings = []
        self.num_tiles = [int(np.ceil((state_high[i] - state_low[i]) / tile_width[i]))
                          for i in range(self.num_state_dimension)]

        self.tiling_size = np.prod(self.num_tiles) * self.num_actions  # tiling_size = number of tiles per tiling

        for i in range(num_tilings):
            tiling = []
            for d_i in range(self.num_state_dimension):
                offset = i * tile_width[d_i] / num_tilings
                buckets = [state_low[d_i] - offset + tile_width[d_i] * k for k in range(self.num_tiles[d_i] + 1)]
                tiling.append(tuple(buckets[1:-1]))
            self.tilings.append(tuple(tiling))

        self.tilings = tuple(self.tilings)
        self.feature_len = self.num_tilings * self.tiling_size

        self.offsets = []
        # calculate offsets for each dimension used for decoding tile later
        k = self.num_actions
        for num_tile in self.num_tiles[::-1]:
            self.offsets.insert(0, k)
            k *= num_tile
        self.offset = tuple(self.offsets)

    def feature_vector_len(self):
        return self.feature_len

    def __call__(self, obs, done, act) -> np.array:
        activated = np.zeros(self.feature_vector_len())
        if not done:
            index = 0
            for tiling_pos in range(self.num_tilings):
                offset = 0
                for d_i in range(self.num_state_dimension):
                    offset += np.digitize(obs[d_i], self.tilings[tiling_pos][d_i]) * self.offsets[d_i]

                activated[index + offset + np.argmax(act)] = 1
                index += self.tiling_size
        return activated


class FixedRandomPolicyAgentTrainer(AgentTrainer):

    def __init__(self, agent_name, observation_shape, action_space):
        self.agent_name = agent_name
        self.action_space = action_space
        self.policy = InboundRandomPolicy(action_space)
        self.w = np.zeros(0)

    def action(self, obs):
        return self.policy.action(obs)

    def update(self, obs, action, reward, next_obs, done):
        pass
    
    def reset(self, obs):
        pass

    def set_weight(self, w):
        self.w = w

class SarsaLambdaAgentTrainer(AgentTrainer):
    def __init__(self, agent_name, observation_shape, action_space, gamma, lam, alpha):
        self.agent_name = agent_name
        self.action_space = action_space
        self.action_space_dim = action_space.n
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.policy = InboundRandomPolicy(action_space)

        # Raw observation has 16 dimensions
        #   agent.state.p_vel 			-- 0, 1                     [min, max] = [-1.3, 1.3] x 1
        #   agent.state.p_pos 			-- 2, 3                     [min, max] = [-1,   1] x 1
        #   entity_pos 		  			-- 4, 5 | 6, 7              [min, max] = [-1,   1] x 2
        #   other_pos		  		    -- 8, 9 | 10, 11 | 12, 13   [min, max] = [-1,   1] x 3
        #   other_vel (good agent vel)	-- 15, 16                   [min, max] = [-1.3, 1.3] x 1

        """Still too slow. need to reduce more"""
        # agent.state.p_vel     -> 2
        # agent.state.p_pos     -> 2
        # distance to two entities -> 2
        # distance to three other agents -> 3
        # state_low = np.array([-1.3, -1.3, -1, -1, -1, -1, -1, -1, -1])
        # state_high = np.array([1.3, 1.3, 1, 1, 1, 1, 1, 1, 1])
        # tile_width = np.array([0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # num_tiling = 1

        state_low = np.array([-1, -1, -1, -1, -1, -1, -1])
        state_high = np.array([1, 1, 1, 1, 1, 1, 1])
        tile_width = np.array([0.25, 0.25, 0.4, 0.4, 0.4, 0.4, 0.4])
        num_tiling = 1

        self.X = TileCoding(state_low, state_high, action_space.n, num_tiling, tile_width)
        self.w = np.zeros((self.X.feature_vector_len()))
        self.a = None
        self.x = None
        self.z = None
        self.Q_old = None

    def reduce_obs_dimensons(self, obs):
        agent_vel = [] # obs[0:2]
        agent_pos = obs[2:4]
        entity_one_pos = obs[4:6]
        entity_two_pos = obs[6:8]
        other_agent_one_pos = obs[8:10]
        other_agent_two_pos = obs[10:12]
        other_agent_three_pos = obs[12:14]

        reduced_ops = np.concatenate([agent_vel] +
                                     [agent_pos] +
                                     [[np.hypot(entity_one_pos[0], entity_one_pos[1])]] +
                                     [[np.hypot(entity_two_pos[0], entity_two_pos[1])]] +
                                     [[np.hypot(other_agent_one_pos[0], other_agent_one_pos[1])]] +
                                     [[np.hypot(other_agent_two_pos[0], other_agent_two_pos[1])]] +
                                     [[np.hypot(other_agent_three_pos[0], other_agent_three_pos[1])]]
                                     )
        return reduced_ops

    def epsilon_greedy_policy(self, obs, done, epsilon=.0):
        Q = np.array([np.dot(self.w, self.X(obs, done, a)) for a in range(self.action_space_dim)])
        u = np.zeros(self.action_space.n)

        if np.random.rand() < epsilon:
            u[np.random.randint(0, self.action_space.n)] = 1
        else:
            u[np.random.choice(np.flatnonzero(Q == Q.max()))] = 1
        return u

    def action(self, obs):
        return self.a

    def update(self, obs, action, reward, next_obs, done):
        obs = self.reduce_obs_dimensons(obs)
        a_next = self.epsilon_greedy_policy(obs, done)
        x_next = self.X(obs, done, a_next)
        Q = np.dot(self.w, self.x)
        Q_next = np.dot(self.w, x_next)
        delta = reward + self.gamma * Q_next - Q
        self.z = self.gamma * self.lam * self.z + (1 - self.alpha * self.gamma * self.lam * np.dot(self.z, self.x)) * self.x
        self.w = self.w + self.alpha * (delta + Q - self.Q_old) * self.z - self.alpha * (Q - self.Q_old) * self.x
        self.Q_old = Q_next
        self.x = x_next
        self.a = a_next

    def reset(self, obs):
        obs = self.reduce_obs_dimensons(obs)

        self.a = self.epsilon_greedy_policy(obs, False)
        self.x = self.X(obs, False, self.a)
        self.z = np.zeros((self.X.feature_vector_len()))
        self.Q_old = 0.

    def set_weight(self, w):
        self.w = w

def SarsaLambda(
        env: MultiAgentEnv,
        gamma: float,  # discount factor
        lam: float,  # decay rate
        alpha: float,  # step size,
        num_episode: int,
        max_episode_len=100,
        use_trained_weight=False
):
    fieldnames = ['Episode',
                  'Average Reward',
                  'Agents Reward',
                  'Episode Reward',
                  'Episode Adversary Reward',
                  'Episode Good Agent Reward']

    rows = [] # for saving to csv file
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/sarsa_out"
    episode_rewards = []  # sum of rewards for all agents
    adversary_rewards = []  # sum of rewards for adversary agents
    good_agent_rewards = []  # sum of rewards for good agents

    # initialize trainers
    trainers = []
    for i, agent in enumerate(env.agents):
        if agent.adversary:
            trainer = SarsaLambdaAgentTrainer("agent_%d" % i, env.observation_space[i], env.action_space[i],
                                                    gamma, lam, alpha)
            if use_trained_weight:
                w = np.load(dir_path + "/agent_%d" % i + "_weight.npy")
                trainer.set_weight(w)
        else:
            trainer = FixedRandomPolicyAgentTrainer("agent_%d" % i, env.observation_space[i], env.action_space[i])

        trainers.append(trainer)

    for i_episode in range(1, num_episode+1):
        ts1 = time.time()
        obs_n = env.reset()
        episode_rewards.append(0.)
        adversary_rewards.append(0.)
        good_agent_rewards.append(0.)
        agent_rewards = np.zeros(len(trainers))

        for i, trainer in enumerate(trainers):
            trainer.reset(obs_n[i])

        for t in range(max_episode_len):

            # get action from each agent's policy
            act_n = []
            for i, trainer in enumerate(trainers):
                act_n.append(trainer.action(obs_n[i]))

            # step
            obs_n, reward_n, done_n, info_n = env.step(act_n)

            for i, trainer in enumerate(trainers):
                trainer.update(obs_n[i], act_n[i], reward_n[i], None, done_n[i])

            # render
            # env.render()
            if i_episode % 10 == 0:
                env.render()

            for i, agent in enumerate(env.agents):
                episode_rewards[-1] += reward_n[i]
                if agent.adversary:
                    adversary_rewards[-1] += reward_n[i]
                else:
                    good_agent_rewards[-1] += reward_n[i]

            agent_rewards += reward_n

        print('\rEpisode {}'
              '\tAverage Reward: {:.2f}'
              '\tAgents Reward: {}'
              '\tEpisode Reward: {:.2f}'
              '\tEpisode Adversary Reward: {:.2f}'
              '\tTime:{}\n'
              .format(i_episode,
                      np.mean(episode_rewards),
                      agent_rewards,
                      episode_rewards[-1],
                      adversary_rewards[-1],
                      time.time() - ts1), end="")

        vals = [i_episode, np.mean(episode_rewards), agent_rewards, episode_rewards[-1], adversary_rewards[-1], good_agent_rewards[-1]]
        rows.append(dict(zip(fieldnames, vals)))

    with open('sarsa_out/benchmark.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for trainer in trainers:
        np.save(dir_path + "/" + trainer.agent_name + "_weight", trainer.w)

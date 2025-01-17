import numpy as np
import random
import copy
from collections import namedtuple, deque

from models import Actor, Critic
from sarsa import FixedRandomPolicyAgentTrainer

import torch
import torch.nn.functional as F
import torch.optim as optim

from multiagent.environment import MultiAgentEnv
from multiagent.policy import Policy
import numpy as np
import time
import os
import csv

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DdpgAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_name, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.agent_name = agent_name
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, )
    
    def update(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def action(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self, obs):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def DDPG(
    env: MultiAgentEnv,
    num_episode: int,
    max_episode_len=100
):
    print('=== Start DDPG ====')
    fieldnames = ['Episode',
                  'Average Reward',
                  'Agents Reward',
                  'Episode Reward',
                  'Episode Adversary Reward',
                  'Episode Good Agent Reward']

    rows = [] # for saving to csv file
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/ddpg_out"
    episode_rewards = []  # sum of rewards for all agents
    adversary_rewards = []  # sum of rewards for adversary agents
    good_agent_rewards = []  # sum of rewards for good agents


    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    trainers = []

    for i, agent in enumerate(env.agents):
        if agent.adversary:
            trainers.append(DdpgAgent("agent_%d" % i, env.observation_space[i].shape[0], env.action_space[i].n))
        else:
            trainers.append(FixedRandomPolicyAgentTrainer("agent_%d" % i, env.observation_space[i], env.action_space[i]))


    for i_episode in range(1, num_episode+1):
        ts1 = time.time()
        state = env.reset()
        for trainer in trainers:
            trainer.reset(state)
        score = np.zeros(len(trainers))
        episode_rewards.append(0.)
        adversary_rewards.append(0.)
        good_agent_rewards.append(0.)
        agent_rewards = np.zeros(len(trainers))

        for t in range(max_episode_len):
            actions = []
            for idx, trainer in enumerate(trainers):
                actions.append(trainer.action(state[idx]))

            next_state, reward, done, _ = env.step(actions)

            if i_episode % 1000 == 0:
                env.render()
            for idx, trainer in enumerate(trainers):
                trainer.update(state[idx], actions[idx], reward[idx], next_state[idx], done[idx])
            state = next_state
            score += reward

            for i, agent in enumerate(env.agents):
                episode_rewards[-1] += reward[i]
                if agent.adversary:
                    adversary_rewards[-1] += reward[i]
                else:
                    good_agent_rewards[-1] += reward[i]
            agent_rewards += reward

            if all(done):
                print("Done with episode:", i_episode)
                break 
        
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
        # if i_episode % 10 == 0:
        #     env.render()
        #     #torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        #     #torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        vals = [i_episode, np.mean(episode_rewards), agent_rewards, episode_rewards[-1], adversary_rewards[-1], good_agent_rewards[-1]]
        rows.append(dict(zip(fieldnames, vals)))

    with open('ddpg_out/benchmark.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
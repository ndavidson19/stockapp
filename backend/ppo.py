
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gym



class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Line 6 of pseudocode: calculate the return
        ## Here, we calculate discounts for instance [0.99^1, 0.99^2, 0.99^3, ..., 0.99^len(rewards)]
        discounts = [gamma**i for i in range(len(rewards)+1)]
        ## We calculate the return by sum(gamma[t] * reward[t]) 
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        # Line 7:
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Line 8:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
    return scores



        
def evaluate_agent(env, max_steps, n_eval_episodes, policy):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param policy: The Reinforce agent
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    
    for step in range(max_steps):
      action, _ = policy.act(state)
      new_state, reward, done, info = env.step(action)
      total_rewards_ep += reward
        
      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

class RLAgent:
    def __init__(self, env, hyperparameters):
        self.env = env
        self.hyperparameters = hyperparameters
        self.policy = Policy(hyperparameters["state_space"], hyperparameters["action_space"], hyperparameters["h_size"]).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=hyperparameters["lr"])
        self.scores = []
        self.mean_rewards = []
        self.std_rewards = []

    def train(self):
        self.scores = reinforce(self.policy,
                                self.optimizer,
                                self.hyperparameters["n_training_episodes"], 
                                self.hyperparameters["max_t"],
                                self.hyperparameters["gamma"], 
                                1000)
        return self.scores

    def evaluate(self):
        mean_reward, std_reward = evaluate_agent(self.env, self.hyperparameters["max_t"], self.hyperparameters["n_evaluation_episodes"], self.policy)
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)
        return mean_reward, std_reward

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    
    def plot_evaluation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.errorbar(np.arange(len(self.mean_rewards)), self.mean_rewards, yerr=self.std_rewards)
        plt.ylabel('Mean reward')
        plt.xlabel('Episode #')
        plt.show()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
    
    def get_policy(self):
        return self.policy
    
    


  '''
  trading_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 20000,
    "n_evaluation_episodes": 10,
    "max_t": 5000,
    "gamma": 0.99,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}


trading_policy = Policy(pong_hyperparameters["state_space"], pong_hyperparameters["action_space"], pong_hyperparameters["h_size"]).to(device)
trading_optimizer = optim.Adam(pong_policy.parameters(), lr=pong_hyperparameters["lr"])
scores = reinforce(pong_policy,
                   pong_optimizer,
                   pong_hyperparameters["n_training_episodes"], 
                   pong_hyperparameters["max_t"],
                   pong_hyperparameters["gamma"], 
                   1000)
  
  '''
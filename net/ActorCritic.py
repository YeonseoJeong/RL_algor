## ActorCritic for SAC

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim =64, std_bound = (1e-2, 1.0)):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound   
        self.std_min, self.std_max = std_bound

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        mu = torch.tanh(mu) * self.action_bound  # Scale to action bounds
        
        log_std = self.std(x)
        log_std = torch.clamp(log_std, math.log(self.std_min), math.log(self.std_max))  # Clamp std to log space
        std = torch.exp(log_std)   

        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()                          # Reparameterization trick a = mu + std * epsilon ~ N(0, 1)
        action = torch.tanh(z) * self.action_bound
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_value(x)
        return q_value
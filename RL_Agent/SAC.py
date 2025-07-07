import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from RL_Agent.utils.ReplayBuffer import ReplayBuffer
from net.ActorCritic import Actor, Critic
'''
Lcritic, Lactor, Lalpha(선택)
y = r + gammma * min(Q1(s', a'), Q2(s', a')) - alpha * log(pi(a'|s'))
Lcritic = MSE(Q1(s, a), y) + MSE(Q2(s, a), y)
Lactor = -min(Q1(s, pi(s)), Q2(s,a)) + alpha * log(pi(a|s)) 
'''

class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, device=None,
                 actor_lr = 3e-4, critic_lr = 3e-4,
                 alpha =0.2, tau = 5e-3, gamma = 0.99):
        self.device = torch.device('cpu') if device is None else device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer(100000)  # Initialize replay buffer with a capacity of 1 million

        self.actor = Actor(state_dim, action_dim, action_bound).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.actor(state)
            print(f"DBG: mu={mu}, std={std}")  # Debugging line to check mu and std
            if torch.isnan(mu).any() or torch.isnan(std).any():
                print(f"⚠️ NaN detected in mu/std:\n  mu={mu}\n  std={std}")
                raise ValueError("NaN in policy parameters")

            if evaluate:
                action = mu
            else:
                action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def update(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(-1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next = self.critic1_target(next_state, next_action)
            q2_next = self.critic2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            td_target = reward + (1 - done) * self.gamma * q_next

        q1_pred = self.critic1(state, action)
        q2_pred = self.critic2(state, action)
        critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_new, log_prob_new = self.actor.sample(state)
        q1_new = self.critic1(state, action_new)
        q2_new = self.critic2(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob_new - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()
    
    def train(self, env, episodes, batch_size, max_steps=1000):
        rewards =[]

        for episode in range(episodes):
            state, _ = env.reset()
            ep_reward = 0
            critic_loss, actor_loss = 0, 0

            for _ in range(max_steps):
                if len(self.replay_buffer) < batch_size:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)
                next_state, reward, done, truns, _ = env.step(action)
                done = done or truns
                self.replay_buffer.put((state, action, reward, next_state, done))
                state = next_state
                ep_reward += reward

                if len(self.replay_buffer) >= batch_size:
                    critic_loss, actor_loss = self.update(batch_size)

                if done:
                    break
                    
            rewards.append(ep_reward)                
            print(f'Episode {episode + 1}, Total Reward: {ep_reward:.2f}, Critic Loss: {critic_loss:.3f}, Actor Loss: {actor_loss:.3f}')
        return rewards
    
    def plot_training(rewards, eval_scores=None, window=10):
        plt.figure(figsize=(10,5))
        # episode rewards
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(len(smoothed)), smoothed, label='Episode Reward (smoothed)')

        # evaluation scores
        if eval_scores:
            eps, scores = zip(*eval_scores)
            plt.plot(eps, scores, 'ro-', label='Eval Score')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('SAC Training')
        plt.tight_layout()
        plt.show()
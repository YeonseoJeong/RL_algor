import gymnasium as gym
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from RL_Agent.SAC import SACAgent

if __name__ == "__main__":
    env = gym.make('Pendulum-v1', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    agent = SACAgent(state_dim, action_dim, action_bound)

    episodes = 200
    batch_size = 128
    max_steps = 200
    rewards = agent.train(env, episodes, batch_size, max_steps)

    agent.plot_training(rewards, window=10)
    env.close()
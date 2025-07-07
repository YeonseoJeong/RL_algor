import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from utils.ReplayBuffer import EpisodeMemory, EpisodeBuffer
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

class Qnet(nn.Module):
    def __init__(self, state_space=None, action_space=None):
        super(Qnet, self).__init__()

        assert state_space is not None, "State space must be defined"
        assert action_space is not None, "Action space must be defined"

        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.fc1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.fc1(x))
        x, (new_h, new_c) = F.relu(self.lstm(x, (h, c)))
        x = self.fc2(x)
        return x, new_h, new_c
    
    def sample_action(self, obs, h, c, epsilon):
        output = self.forward(obs, h, c)

        if random.random() < epsilon:
            return random.randint(0,1), output[1], output[2]
        else:
            action = output[0].argmax(dim=1).item()
            return action, output[1], output[2]
        
    def init_hidden_state(self, batch_size, training = None):

        assert training is not None, "Training mode must be specified"

        if training:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1,1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])
    

def train(q_net=None, target_q_net=None, episode_memory=None,
          device=None, 
          optimizer = None,
          batch_size=1,
          learning_rate=1e-3,
          gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(observations.reshape(batch_size,seq_len,-1)).to(device)
    actions = torch.LongTensor(actions.reshape(batch_size,seq_len,-1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size,seq_len,-1)).to(device)
    next_observations = torch.FloatTensor(next_observations.reshape(batch_size,seq_len,-1)).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size,seq_len,-1)).to(device)

    h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)

    q_target, _, _ = target_q_net(next_observations, h_target.to(device), c_target.to(device))

    q_target_max = q_target.max(2)[0].view(batch_size,seq_len,-1).detach()
    targets = rewards + gamma*q_target_max*dones


    h, c = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, _, _ = q_net(observations, h.to(device), c.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss        
    loss = F.smooth_l1_loss(q_a, targets)
    
    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":

    # Env parameters
    model_name = "DRQN_POMDP_Random"
    env_name = "CartPole-v1"
    seed = 1
    exp_num = 'SEED'+'_'+str(seed)

    # Set gym environment
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    # env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+env_name+"_"+model_name+"_"+exp_num)

    # Set parameters
    batch_size = 8
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_epi_num = 20 # Start moment to train the Q network
    episodes = 650
    print_per_iter = 20
    target_update_period = 4
    eps_start = 0.1
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 2000

    # DRQN param
    random_update = True# If you want to do random update instead of sequential update
    lookup_step = 20 # If you want to do random update instead of sequential update
    max_epi_len = 100 
    max_epi_step = max_step

    

    # Create Q functions
    Q = Qnet(state_space=env.observation_space.shape[0]-2, 
              action_space=env.action_space.n).to(device)
    Q_target = Qnet(state_space=env.observation_space.shape[0]-2, 
                     action_space=env.action_space.n).to(device)

    Q_target.load_state_dict(Q.state_dict())

    # Set optimizer
    score = 0
    score_sum = 0
    optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

    epsilon = eps_start
    
    episode_memory = EpisodeMemory(random_update=random_update, 
                                   max_epi_num=100, max_epi_len=600, 
                                   batch_size=batch_size, 
                                   lookup_step=lookup_step)

    # Train
    for i in range(episodes):
        s, info = env.reset(seed=seed) # Reset environment
        obs = s[::2] # Use only Position of Cart and Pole
        done = False
        
        episode_record = EpisodeBuffer()
        h, c = Q.init_hidden_state(batch_size=batch_size, training=False)

        for t in range(max_step):

            # Get action
            a, h, c = Q.sample_action(torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0), 
                                              h.to(device), c.to(device),
                                              epsilon)

            # Do action
            s_prime, r, done, truncated, info = env.step(a)
            obs_prime = s_prime[::2]

            # make data
            done_mask = 0.0 if done else 1.0

            episode_record.put([obs, a, r/100.0, obs_prime, done_mask])

            obs = obs_prime
            
            score += r
            score_sum += r

            if len(episode_memory) >= min_epi_num:
                train(Q, Q_target, episode_memory, device, 
                        optimizer=optimizer,
                        batch_size=batch_size,
                        learning_rate=learning_rate)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): # <- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        
        episode_memory.put(episode_record)
        
        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        if i % print_per_iter == 0 and i!=0:
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            i, score_sum/print_per_iter, len(episode_memory), epsilon*100))
            score_sum=0.0
            save_model(Q, model_name+"_"+exp_num+'.pth')

        # Log the reward
        writer.add_scalar('Rewards per episodes', score, i)
        score = 0
        
    writer.close()
    env.close()
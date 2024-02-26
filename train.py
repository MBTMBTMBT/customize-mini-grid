import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from minigrid.wrappers import FullyObsWrapper
from custom_env import CustomEnvFromFile


# DQN Agent
class DQNAgent:
    def __init__(self, state_space, action_space, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=10000)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )
        return model

    def act(self, state):
        if state.ndim == 1:
            state = torch.FloatTensor(state).unsqueeze(0)
        else:
            state = torch.FloatTensor(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        action_values = self.model(state)
        return np.argmax(action_values.detach().numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).squeeze(1)  # Adjusted to squeeze the states
        actions = torch.LongTensor(actions).view(-1, 1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states).squeeze(1)  # Similarly, adjust next_states
        dones = torch.BoolTensor(dones)

        # Compute Q values for current states
        output = self.model(states)
        Q_values = self.model(states).gather(1, actions)

        # Compute V values for next states using target network
        next_Q_values = self.model(next_states).detach()
        max_next_Q_values = next_Q_values.max(1)[0].unsqueeze(1)
        expected_Q_values = rewards + (self.gamma * max_next_Q_values * (~dones))

        # Compute loss
        loss = F.mse_loss(Q_values, expected_Q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)  # Clamping the gradients to stabilize learning
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay


def preprocess_observation(obs):
    # Extract the 'image' part of the observation and flatten it
    image_obs = obs['image']
    return np.ravel(image_obs)


if __name__ == "__main__":
    env = FullyObsWrapper(CustomEnvFromFile(txt_file_path='test.txt'))
    # Assuming you're focusing on the 'image' part for your DQN input
    image_shape = env.observation_space.spaces['image'].shape
    state_space = np.prod(image_shape)
    action_space = env.action_space.n

    agent = DQNAgent(state_space, action_space)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        obs, _ = env.reset()  # Discarding info as it's not used here
        state = preprocess_observation(obs)
        state = np.reshape(state, [1, state_space])
        for time in range(500):
            action = agent.act(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(next_obs)
            next_state = np.reshape(next_state, [1, state_space])
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

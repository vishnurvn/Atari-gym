import ale_py
import gymnasium as gym
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from models import SimplePolicy

env = gym.make("ALE/Assault-v5")
action_space = 7
num_episodes = 10_000
BATCH_SIZE = 32

policy = SimplePolicy(action_space)
optimizer = optim.Adam(policy.parameters(), lr=0.001)


class TrajectoryData(Dataset):
    def __init__(self, observations, rewards, actions) -> None:
        super().__init__()
        self.observations = observations
        self.actions = actions
        self.reward = sum(rewards) / len(rewards)

    def __getitem__(self, index):
        return self.observations[index], self.actions[index], self.reward

    def __len__(self):
        return len(self.observations)


def obs_transform(obs):
    obs = torch.tensor(obs) / 255.0
    obs = obs.permute(2, 0, 1)
    return obs.unsqueeze(0)


def compute_loss(observations, weights, actions):
    return -(policy(observations).log_prob(actions) * weights).mean()


train_env = gym.wrappers.TransformObservation(env, obs_transform, env.observation_space)

for ep in range(num_episodes):
    done = False
    state, *_ = train_env.reset()

    episode_rewards = []
    episode_observations = []
    episode_actions = []

    while not done:
        action = policy.get_action(state).sample().item()
        state, reward, terminated, truncated, info = train_env.step(action)

        episode_observations.append(state)
        episode_rewards.append(reward)
        episode_actions.append(action)

        done = terminated or truncated

    observations = torch.vstack(episode_observations)
    rewards = torch.as_tensor(episode_rewards)
    actions = torch.as_tensor(episode_actions)
    episode_loss = 0

    optimizer.zero_grad()
    trajectory_ds = TrajectoryData(observations, rewards, actions)
    loader = DataLoader(trajectory_ds, batch_size=32, shuffle=False)

    for obs, acts, wts in loader:
        loss = compute_loss(obs, wts, acts)
        loss.backward()
        episode_loss += loss
    optimizer.step()

    print(
        f"End of ep: {ep}, episode length: {len(episode_rewards)}, loss: {episode_loss}, reward: {sum(episode_rewards)}"
    )

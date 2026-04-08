import ale_py
import gymnasium as gym

env = gym.make("ALE/Assault-v5")
env.reset()

done = False
while not done:
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    done = trunc or term
    print(obs.min())

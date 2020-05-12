import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import gym
env = gym.make('CartPole-v0')



state1 = env.reset()
action = env.action_space.sample()
state, reward, done, info = env.step(action)
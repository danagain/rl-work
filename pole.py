#import sys
#sys.path.append('/usr/local/lib/python3.7/site-packages')
import gym
env = gym.make('CartPole-v0')
import numpy as np
import torch
from matplotlib import pyplot as plt

# Open AI gym
state1 = env.reset()
action = env.action_space.sample()
state, reward, done, info = env.step(action)

# Create policy NN
l1 = 4
l2 = 150
l3 = 2

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax()
)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#pred = model(torch.from_numpy(state1).float()) # Feed the input state through the network to get policy prediction
#action = np.random.choice(np.array([0,1]), p=pred.data.numpy()) # The action we choose is based on the probability output by the policy network
#state2, reward, done, info = env.step(action) # get the next state, reward, done flag and info from AI gym

# Computing the discounted rewards
def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return

def loss_fn(preds, r):
    return -1 * torch.sum(r * torch.log(preds))


epochs = []
episode_duration = []

MAX_DUR = 200
MAX_EPISODES = 1000
gamma = 0.99
score = []

for episode in range(MAX_EPISODES):
    curr_state = env.reset()
    done = False
    transitions = []

    for t in range(MAX_DUR):
        act_prob = model(torch.from_numpy(curr_state).float()) # Feed the current state through the policy network to get action predicitons
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy()) # stochastically select the next action 0,1 (left right)
        prev_state = curr_state # prepare to generate the new state after taking action
        curr_state, _, done, info = env.step(action) # take the action and move into the next state
        transitions.append((prev_state, action, t+1)) # store the transition information (need the action probability value and the t step for loss function)
        if done: # if the move we just took ended the game, break this loop
            epochs.append(episode)
            episode_duration.append(len(transitions))
            break
    ep_len = len(transitions) # how many actions have we done in the episode
    score.append(ep_len) # keep track of the episode length over training
    reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))
    disc_rewards = discount_rewards(reward_batch) # Calculate the discounted rewrads for the episode. The reward is +1 for every action e.g 1, 2, 3, 4, 5  .. state 5 has reward = 5, these values are exponentially decayed to make the 
    # last few actions more important .. this is because the last few actions in the game are more impactful in losing the game (pole falling over)
    state_batch = torch.Tensor([s for (s, a, r) in transitions]) # collect the states in the episode into a single Tensor
    action_batch = torch.Tensor([a for (s, a, r) in transitions]) # collect the actions int the episode into a single Tensor 
    pred_batch = model(state_batch)
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze() # subsets the action-probabilities associated with the actions that were actually taken
    loss = loss_fn(prob_batch, disc_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.figure(figsize=(10, 7))
plt.scatter(epochs, episode_duration)
plt.xlabel("Epochs", fontsize=22)
plt.ylabel("Episode Duration",  fontsize=22)
plt.show()





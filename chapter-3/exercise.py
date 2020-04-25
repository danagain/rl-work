from Gridworld import Gridworld
import numpy as np
import torch
import random
from matplotlib import pylab as plt
from collections import deque
from IPython.display import clear_output
import copy
'''
Creating the QLearning Neural network
'''
# NN layer sizes
l1 = 64
l2 = 150
l3 = 100
l4 = 4

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
)

model2 = copy.deepcopy(model) # Create a copy of the neural network to create the target network
model2.load_state_dict(model.state_dict()) # copy the parameters from the original model
# copy the params of model into model2
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gamma = 0.9
epsilon = 0.3

game = Gridworld(size=4, mode='static')
game.display()

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

'''
Setting up the main training loop
'''
epochs = 5000
sync_freq = 500 # variable to sync the target and original neural nets, evert 50 steps we will
losses = [] # Create a list to store loss values so we can plot the trend later
mem_size = 1000 # set the total size of the experience replay memory
batch_size = 200 # set the mini batch size
replay = deque(maxlen=mem_size)
max_moves = 50 # max number of moves before the game is over
h = 0
j = 0
plt.xlabel("Iterations")
plt.ylabel("Loss value")
for i in range(epochs):
    game = Gridworld(size=4, mode='random') # create a new game with random placement of the player, wall and trap
    state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0 # add some noise to the input as
    # we are using relu acitvation
    state1 = torch.from_numpy(state1_).float() # get the input vector into a torch tensor
    status = 1 # while loop exit condition
    mov = 0 # number of player moves
    while (status == 1):
        j += 1 # update the sync counter
        mov += 1 # update the player move counter
        qval = model(state1)  # get the qvalues for the current state
        qval_ = qval.data.numpy() # turn into numpy array
        if (random.random() < epsilon):  # epsilon greedy method
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_] # get the action character 'u', 'd', 'l', 'r'
        game.makeMove(action) # make the move
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0 # get the input vector of the
        # new state
        state2 = torch.from_numpy(state2_).float() # turn into torch tensor
        reward = game.reward() # get the reward amount for the new state
        done = True if reward > 0 else False # determine if the game is finished
        exp = (state1, action_, reward, state2, done)  # record replay experience for batch processing
        replay.append(exp)  # add the recorded replay experience to the deque list
        state1 = state2 # update the new state

        if len(replay) > batch_size:  # if we hit the max limit on the replay buffer
            minibatch = random.sample(replay, batch_size)  # take a random sample of the replay buffer
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])  # create the tensor batches
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            Q1 = model(state1_batch)  # batch calculation of the q values
            with torch.no_grad(): # exit computation graph context to calc new state St+1 Q value to find expected value
                # of A in St
                Q2 = model2(state2_batch)  # Batch of Q values for St+1 - value of moving from S to St+1 determined
                # by Q values for state St+1

            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])  # Calculate the 'value'
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze() # Get what the network
            # currently thinks is the value
            loss = loss_fn(X, Y.detach()) # run the loss function MSE based on the how much the calculated expected
            # 'value' differs from the current value
            print(i, loss.item())
            clear_output(wait=True)
            optimizer.zero_grad() # perform gradient decent
            loss.backward() # perform back propagation
            losses.append(loss.item()) # append the loss value to a list for graphing purposes
            optimizer.step() # perform a gradient descent step
            if j % sync_freq == 0: # if we reach the sync iteration, sync the target network and our model
                model2.load_state_dict(model.state_dict()) # sync across weights

        if reward != -1 or mov > max_moves:  # if we have WON/LOST or played to many steps
            status = 0 # exit the while loop
            mov = 0 # reset the moves
losses = np.array(losses) # record the losses


'''
Testing the solution
'''


def test_model(model, mode='random', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    while (status == 1):  # A
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)  # B
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win


max_games = 10
wins = 0
for i in range(max_games):
    win = test_model(model, mode='random', display=True)
    if win:
        wins += 1
win_perc = float(wins) / float(max_games)
print("Games played: {0}, # of wins: {1}".format(max_games, wins))
print("Win percentage: {}%".format(100.0*win_perc))

plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Epochs", fontsize=22)
plt.ylabel("Loss",  fontsize=22)
plt.show()
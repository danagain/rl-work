import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


# work out the reward we get, the average reward returned from this will be dependant on how good the average mean generated is
def get_reward(prob, n=10):
    reward = 0
    for i in range(n): # for the range of $0 to $10
        if random.random() < prob: # if the random float is less than the mean probability then bump reward value
            reward += 1 # lower avg mean probabilities will need more luck to get a reward of 10
    return reward

# Function to update a new record
def update_record(record,action,r):
    # Update the mean average.. because mean avg is:  sum(Q for action k) / number of times action k we can do
    # Number of actions * sum(Q for action k) to get back to the sum then add the new action and divide again by num actions + 1 to recalc the mean avg
    new_r = (record[action,0] * record[action,1] + r) / (record[action,0] + 1) # Calculate an updated mean average value
    record[action,0] = record[action,0] + 1
    record[action,1] = new_r
    return record

def get_best_arm(record):
    arm_index = np.argmax(record[:,1],axis=0) # get the biggest Q value out of all the machines : 
    return arm_index

'''
Now we can get into the main loop for playing the n-armed bandit game. 
If a random number is greater than the epsilon parameter, we just calculate the best action
 using the get_best_arm function and take that action. Otherwise we take a random action 
 to ensure some amount of exploration. After choosing the arm, we use the get_reward function 
 and observe the reward value. We then update the record array with this new observation. 
 We repeat this process a bunch of times, and it will continually update the record array. 
 The arm with the highest reward probability should eventually get chosen most often, since 
 it will give out the highest average reward. We’ve set it to play 500 times in the following 
 listing, and to display a matplotlib scatter plot of the mean reward against plays. Hopefully 
 we’ll see that the mean reward increases as we play more times.
'''

print(np.mean([get_reward(0.7) for _ in range(2000)]))
n = 10 # Number of slot machines
probs = np.random.rand(n) # Mean probability for each slot machine. E.g 0.7 will have probability distribution highest at $7 reward
eps = 0.2 # Epsilon value - 20 percent chance a random action will be performed. 70 percent chance we will take the highest Q value action
record = np.zeros((n,2)) # this record is an n x 2 matrix which will hold the mean average reward and the number of times the machine has been played for each machine
print(record)
fig, ax = plt.subplots(1,1) 
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward") 

rewards = [0]
for i in range(500):
    if random.random() > eps:
        choice = get_best_arm(record)
    else:
        choice = np.random.randint(10)
    r = get_reward(probs[choice])
    record = update_record(record,choice,r)
    mean_reward = ((i+1) * rewards[-1] + r) / (i+2) # i + 1 to deal with i = 0. Keeping track of mean reward
    rewards.append(mean_reward)
ax.scatter(np.arange(len(rewards)),rewards)
plt.xlabel('Plays')
plt.ylabel('Avg Reward')
plt.show()


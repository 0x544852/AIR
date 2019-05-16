import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time
from SlotGame import SlotGame


seed = 1546847731  # or try a new seed by using: seed = int(time())
random.seed(seed)


# print('Seed: {}'.format(seed))


def state_to_str(state):
    return str(list(map(int, state.tolist())))


all_states = list()
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                s = np.array([i, j, k, l])
                all_states.append(state_to_str(s))

#print('All possible states:')
#for s in all_states:
#    print(s)

game = SlotGame()

num_of_games = 3000
epsilon = 0.25
gamma = 1

q_table = pd.DataFrame(0, index=np.arange(4), columns=all_states)

r_list = []  # store the total reward of each game so we can plot it later

for g in range(num_of_games):
    game_over = False
    game.reset()
    total_reward = 0
    while not game_over:
        state = np.copy(game.board)
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = q_table[state_to_str(state)].idxmax()
        reward, game_over = game.play(action)
        total_reward += reward
        if np.sum(game.board) == 4:  # terminal state
            next_state_max_q_value = 0
        else:
            next_state = np.copy(game.board)
            next_state_max_q_value = q_table[state_to_str(next_state)].max()
        q_table.loc[action,state_to_str(state)] = reward + gamma * next_state_max_q_value
    r_list.append(total_reward)
#pd.set_option('display.max_columns', 16)
#print (q_table)

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                b = np.array([i,j,k,l])
                if len(np.where(b == 0)[0]) != 0:
                    action = q_table[state_to_str(b)].idxmax()
                    pred = q_table[state_to_str(b)].tolist()
                    print('board: {b}\tpredicted Q values: {p} \tbest action: {a}\tcorrect action? {s}'
                          .format(b=b,p=pred,a=action,s=b[action]==0))

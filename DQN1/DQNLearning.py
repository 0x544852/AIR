import random
import numpy as np
import tensorflow as tf
from QNetwork import QNetwork
from ReplayMemory import ReplayMemory
from SlotGame import SlotGame

seed = 1546847731  # or try a new seed by using: seed = int(time())
random.seed(seed)
# print('Seed: {}'.format(seed))

game = SlotGame()

num_of_games = 2000
epsilon = 0.1
gamma = 0.99
batch_size = 10
memory_size = 2000

tf.reset_default_graph()
tf.set_random_seed(seed)
qnn = QNetwork(hidden_layers_size=[20,20], gamma=gamma, learning_rate=0.001, a_seed=seed)
memory = ReplayMemory(memory_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

r_list = []
c_list = []  # same as r_list, but for the cost

counter = 0  # will be used to trigger network training

for g in range(num_of_games):
    game_over = False
    game.reset()
    total_reward = 0
    while not game_over:
        counter += 1
        state = np.copy(game.board)
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            pred = np.squeeze(sess.run(qnn.output,feed_dict={qnn.states: np.expand_dims(game.board,axis=0)}))
            action = np.argmax(pred)
        reward, game_over = game.play(action)
        total_reward += reward
        next_state = np.copy(game.board)
        memory.append(
            {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'game_over': game_over})
        if counter % batch_size == 0:
            # Network training
            batch = memory.sample(batch_size)
            q_target = sess.run(qnn.output,
                                feed_dict={qnn.states: np.array(list(map(lambda x: x['next_state'], batch)))})
            terminals = np.array(list(map(lambda x: x['game_over'], batch)))
            for i in range(terminals.size):
                if terminals[i]:
                    # Remember we use the network's own predictions for the next state while calculatng loss.
                    # Terminal states have no Q-value, and so we manually set them to 0, as the network's predictions
                    # for these states is meaningless
                    q_target[i] = np.zeros(game.board_size)
            _, cost = sess.run([qnn.optimizer, qnn.cost],
                               feed_dict={qnn.states: np.array(list(map(lambda x: x['state'], batch))),
                                          qnn.r: np.array(list(map(lambda x: x['reward'], batch))),
                                          qnn.enum_actions: np.array(
                                              list(enumerate(map(lambda x: x['action'], batch)))),
                                          qnn.q_target: q_target})
            c_list.append(cost)
    r_list.append(total_reward)
    #print('Final cost: {}'.format(c_list[-1]))

for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                b = np.array([i,j,k,l])
                if len(np.where(b == 0)[0]) != 0:
                    pred = np.squeeze(sess.run(qnn.output,feed_dict={qnn.states: np.expand_dims(b,axis=0)}))
                    pred = list(map(lambda x: round(x,3),pred))
                    action = np.argmax(pred)
                    print('board: {b}\tpredicted Q values: {p} \tbest action: {a}\tcorrect action? {s}'
                          .format(b=b,p=pred,a=action,s=b[action]==0))
sess.close()  # Don't forget to close tf.session

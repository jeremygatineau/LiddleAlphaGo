import gym
import copy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from MCTS import MCTS
from model import PolicyNet

GRID = 7

def to_numpy(state): 
    # Takes a state as given by env.step and return the numpy array for the board and the passing indicator
    P0 = state[0] # 2D array of the pieces of P0
    P1 = state[1] # 2D array of the pieces of P1
    turn = state[2][0][0] # =0 if it's P0's turn =1 if P1's
    prev_pass =  state[4][0][0] # Indicicator for whether the previous turn was a pass
    if turn == 0: # Board representation, 1 for the current player's pieces -1 for the other's
        board = P0 - P1
    else :
        board = P1 - P0
    return board, prev_pass

lr=5e-3
nb_data_gather_games=10
max_time_step = 10
temperature = 5
c = 1
batch_size = 32
epochs = 50
tournament_len = 3

env_state = gym.make('gym_go:go-v0', size=GRID, reward_method='real')


MctsD = MCTS(PolicyNet(GRID, 5, GRID**2+1, reg=0.1), 300, GRID)
MctsT = MCTS(PolicyNet(GRID, 5, GRID**2+1, reg=0.1), 300, GRID)

optimizer = tf.optimizers.Adam(lr)
while True:
    HIST = []
    for game_ix in range(nb_data_gather_games):
        print(f"self-play game {game_ix+1}")
        state = env_state.reset()
        t=0
        while t<max_time_step : #and avg_v > min_val:
            t0 = time.time()
            pie, actions = MctsD.search(env_state, temperature, c)
            #print(f"MCTS search took {time.time()-t0} seconds")
            #print(f"pie is {pie}, actions are {actions} for t={t}")
            HIST.append((to_numpy(state), pie, actions, env_state.turn()))
            action = np.random.choice(actions, p=pie)
            state, _, _, _ = env_state.step(action)
            if env_state.game_ended():
                break
            t += 1
        z = 0 if env_state.get_winning()==1 else 1
    #HIST = [[board.flatten(), [ind], pie.flatten(), [int(z==turn_id)]] for (board, ind), pie, turn_id in HIST]
    #print(f"HIST shape is {HIST.shape}")
    print(f"Starting training")

    boardDat = tf.data.Dataset.from_generator(lambda: iter([board.flatten() for (board, _), _, _, _ in HIST]), tf.float32)
    indDat = tf.data.Dataset.from_generator(lambda: iter([[ind] for (_, ind), _, _, _ in HIST]), tf.float32)
    #print(f"pie shape {pie.shape}")
    PL = []
    for _, pie, actions, _ in HIST:
        pl = np.zeros(GRID*GRID+1)
        #print(actions)
        for idd, ix in enumerate(actions.flatten()):
            if ix is None:
                pl[-1] = pie[idd]
            else :
                pl[ix] = pie[idd]
        PL.append(list(pl))
    pieDat = tf.data.Dataset.from_generator(lambda: iter(PL), tf.float32)
    zDat = tf.data.Dataset.from_generator(lambda: iter([[int(z==turn_id)] for _, _, _, turn_id in HIST]), tf.float32) 
    
    dataset = tf.data.Dataset.zip((boardDat, indDat, pieDat, zDat))
    print(dataset)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = list(dataset.as_numpy_iterator())
    #print(len(iterator))
    for epoch in range(epochs):
        #print(f"Training epoch {epoch+1}")
        losses = []
        for batch in iterator:
            #x, y = batch
            #print(f"batch is {batch}")
            
            board, ind, pie, z = batch # Inputs: board as a numpy array and passing indicator (ind)
            #print(f"new : {batch_size*GRID*GRID}, old : {board.shape}")
            board = tf.reshape(board, (batch_size, GRID, GRID, 1))
            ind = tf.reshape(ind, (batch_size, 1))
            # Labels: Target probability distribution (pie) and winning indicator (z)
            #print(f"shapes are board {board.shape}, ind {ind.shape}, pie {pie.shape}, z {z.shape}")
            with tf.GradientTape() as tape:
                #print(f"board shape {board.shape}, ind shape {ind.shape}")
                p, v = MctsT.guidingNet(board, ind, training=True)
                #print(f"pie shape {pie.shape}, p shape {p.shape},  v shape {v.shape}, z shape {z.shape}")
                MSE = tf.losses.MSE(z,v)
                CE = tf.losses.categorical_crossentropy(pie, p)
                loss = MSE+CE# compute the loss
                for lo in MctsT.guidingNet.losses: # Adding the regularization terms for each layer
                    loss+=lo
                losses.append(loss)
                variables = MctsT.guidingNet.trainable_variables # get the trainable variables
                gradients = tape.gradient(loss, variables) # compute the gradients of the loss wrt those variables

            optimizer.apply_gradients(zip(gradients, variables)) # update the trainable weights
        print(f"Mean loss on epoch {epoch+1} is {np.mean(losses)}")
    print(f"starting tournament")
    t_score = 0
    for i in range(tournament_len):
        print(f"Tournament game {i}")
        # Tournamnet to see would wins in a MCTS battle between the training net and the data generation net
        env_state.reset()
        A = ['t', 'd'] if i%2 else ['d', 't'] # to alternate who starts
        D = {'t' : MctsT,  'd' : MctsD}
        t=0
        while t<max_time_step : # and avg_v > min_val:
            for k in A:
                pie, actions = D[k].search(env_state, temperature, c)
                action = np.random.choice(actions, p=pie)
                env_state.step(action)
                if env_state.game_ended():
                    break
                t+=1
            if env_state.game_ended():
                    break

        z = env_state.get_winning()

        if z == 1:
            # A[0] won cuz he started
            if A[0] == 't':
                t_score += 1
            else :
                t_score -= 1
        elif z == -1:
            # A[1] won cuz he didnt start
            if A[1] == 't':
                t_score += 1
            else :
                t_score -= 1
    print(f"tournament ended with a score for the Training net of {t_score}")
    # Here t_score represent the game_won-game_lost for the training network 
    if t_score>0 : 
        # If the training net won more games that means it's better so we use it for the next MCTS data gathering
        MctsD.guidingNet = copy.copy(MctsT.guidingNet)
import numpy as np
import time
import copy

class Node:
    def __init__(self, state, parent):
        self.state = copy.copy(state)
        self.isTerminal = state.game_ended()
        self.parent = parent
        self.V = 0
        self.children = {}    # { action : {'node': NodeObj, 'n'=0, 'w'=0, 'q'=0, 'p'=0...}} } for each edge on that node going to 'state'
    
    def isLeaf(self):
        return len(self.children.keys()) == 0

    def to_numpy(self):
        P0 = self.state.state[0] # 2D array of the pieces of P0
        P1 = self.state.state[1] # 2D array of the pieces of P1
        turn = self.state.state[2][0][0] # =0 if it's P0's turn =1 if P1's
        prev_pass =  self.state.state[4][0][0] # Indicicator for whether the previous turn was a pass
        #print(turn)
        if turn == 0: # Board representation, 1 for the current player's pieces -1 for the other's
            board = P0 - P1
        else :
            board = P1 - P0
        return board, prev_pass

class MCTS():
    def __init__(self, net, timelimit, GRID):
        self.guidingNet = copy.copy(net)
        self.timeLimit = timelimit
        self.GRID = GRID

    def search(self, initState, temperature, c, timeLimit=100):
        self.root = Node(initState, None)
        
        timeLimit = time.time() + self.timeLimit / 1000
        
        while time.time() < timeLimit:
            self.executeRound(c)
        
        ns = np.array([self.root.children[action]['n'] for action in self.root.children.keys()])
        actions = np.array([action for action in self.root.children.keys()])
        #print(f"ns {ns}")
        pie = np.power(ns, 1/temperature)/sum(np.power(ns, 1/temperature))
        return pie, actions

    def executeRound(self, c):
        
        # Selection
        node = self.root
        path = []
        
        while not(node.isLeaf()): #until we find a lead node
            node, action = self.sample(node, c)
            path.insert(0, action)
        
        leaf_player = node.state.turn()
        last_v = node.V
        # Evaluate and expand the leaf node

        node = self.expand(node)

        # Backpropagate to update the values and to keep track of the nodes we have visited
        while True:
            if len(path)==0:
                self.root=node
                break
            action = path.pop(0)
            parent = node.parent
            parent.children[action]['node'] = node
            parent.children[action]['n'] += 1
            if parent.state.turn() == leaf_player: 
                parent.children[action]['w'] += last_v
            else :
                parent.children[action]['w'] -= last_v
            parent.children[action]['q'] = parent.children[action]['w']/parent.children[action]['n']
            
            if parent.parent is None:
                self.root = parent
                break
            node = parent

    def sample(self, node, c):
        pa = np.array([node.children[action]['p'] for action in node.children.keys()]).flatten() # Probability distribution for getting to each child node
        qa = np.array([node.children[action]['q'] for action in node.children.keys()]).flatten()
        na = np.array([node.children[action]['n'] for action in node.children.keys()]).flatten()
        s = np.sum(na)
        
        tm = qa+c*pa*np.sqrt(s)/(1+na)
        
        
        best_action_ix = np.argmax(tm)
        
        best_action = list(node.children.keys())[best_action_ix]
        return node.children[best_action]['node'], best_action
    
    def expand(self, node):
        board, pass_ind = node.to_numpy()
        pa, v = self.guidingNet(board.reshape((1, self.GRID, self.GRID, 1)), pass_ind.reshape((1, 1)), training=False) # pa is a list of probabilities corresponding to the grid flattened and an indicator for whether the previous turn was a pass.
        pa = self.sanitize(pa.numpy(), node.state).flatten()
        #print(f"shape of pa in expand is {pa.shape}")
        node.V = v.numpy()
        for ix, p in enumerate(pa):
            if p != 0 :
                if ix==(len(pa)-1):
                    action = None
                else:
                    action = ix
                nxt_state = copy.copy(node.state)
                nxt_state.step(action)
                newNode = Node(nxt_state, node)
                node.children[action] = {'node': newNode, 'n' : 1, 'w' : 0, 'q' :0, 'p': p}

        return node

    def sanitize(self, pa, state): 
        # Focefully set the probability of invalid moves to 0 and normalize
        playable = state.get_valid_moves()
        ps = pa*playable
        ps[:-1] = ps[:-1]/sum(ps[:-1])
        return ps

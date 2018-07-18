import math
import numpy as np
import torch
from Simulator import Simulator as sim
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, model, state, turn, n_run):
        self.nnet = model
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.canonicalBoard = state
        self.numMCTSSims = n_run
        self.cpuct = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actionsize= 2048
        self.turn = turn


    def getActionProb(self, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.numMCTSSims):
            if i%50 == 0:
                print("i : %d"%i)
            self.search(self.canonicalBoard, self.turn)

        s = self.stringRepresentation(self.canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.actionsize)]
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def stringRepresentation(self, board):
        return board.tostring()

    def coordinates_to_plane(self, coordinates):
        # x: 0.14 4.61
        # y: 11.135 2.906
        if len(coordinates.shape) == 1:
            coordinates = [coordinates]
        number_of_coor = len(coordinates)
        coors = []
        for coordinate in coordinates:
            coors.append([])
            for x, y in zip(coordinate[::2], coordinate[1::2]):
                if x == 0 and y == 0:
                    coors[-1].append(None)
                else:
                    coors[-1].append([int(round((x - 0.14) / 4.47 * 31)), int(round((y - 2.906) / 8.229 * 31))])

        plane = torch.zeros((number_of_coor, 2, 32, 32))
        ones_plane = torch.ones((number_of_coor, 1, 32, 32))
        plane = torch.cat((plane, ones_plane), 1)

        for bat, coor in enumerate(coors):
            for i, c in enumerate(coor):
                if c is None:
                    continue
                x, y = c
                if i % 2 == 0:
                    plane[bat][0][y][x] = 1
                else:
                    plane[bat][1][y][x] = 1

        return plane

    def idx_to_action(self, idx):
        if idx - 1024 < 0:
            turn = 0
        else:
            turn = 1

        if turn == 1:
            tmp = idx - 1024
            rows = tmp // 32
            cols = (idx - 1024) - rows * 32

        else:
            tmp = idx
            rows = tmp // 32
            cols = idx - rows * 32

        x = 4.75 / 31 * cols
        y = 11.28 / 31 * rows
        return [x, y, turn]

    def calculate_value(self, v):
        final_v = 0
        for i in range (-8, 9):
            final_v = final_v + (i * v[i+8])
        return final_v

    def search(self, canonicalBoard, turn):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.stringRepresentation(canonicalBoard)
        '''
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]
        '''

        if s not in self.Ps:
            # leaf node
            state_plane = self.coordinates_to_plane(canonicalBoard).to(self.device)
            self.Ps[s], v = self.nnet(state_plane)
            self.Ps[s] = self.Ps[s][0]
            v = v[0]
            self.Ns[s] = 0

            '''
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            '''
            return v
            #return -v

        # expand 한 상태라면 value값 받아서 바로 보내기 (컬링에서는 Turn이 올라가도 같은 state가 나올 수 있기 때문에 이 부분이 없다면 무한루프에 빠질 수 있음)
        if turn > self.turn:
            state_plane = self.coordinates_to_plane(canonicalBoard).to(self.device)
            self.Ps[s], v = self.nnet(state_plane)
            self.Ps[s] = self.Ps[s][0]
            v = v[0]
            return v

        #valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.actionsize):
            '''
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
             '''

            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        #next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        #next_s = self.game.getCanonicalForm(next_s, next_player)
        action = self.idx_to_action(a)
        next_s = sim.simulate(canonicalBoard, self.turn, action[0], action[1], action[2], 0)[0]
        '''
        expand
        '''
        v = self.search(next_s, turn+1)
        v = self.calculate_value(v)

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        #return -v
        return v

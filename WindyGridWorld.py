import numpy as np
import sys
from gym.envs.toy_text import discrete

UP, RIGHT, DOWN, LEFT = range(4)


class WindyGridWorld(discrete.DiscreteEnv):
    def __init__(self):
        self.shape = (7, 10)
        nS = self.shape[0] * self.shape[1]
        nA = 4
        winds = np.zeros(self.shape)
        winds[:, [3,4,5, 8]] = 1
        winds[:, [6, 7]] = 2
        self.goal = (3, 7)
        # Transition probability calculation from GridWorld
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)            
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)            
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)            
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
        # Starting position (3, 0)
        initial_s = np.zeros(nS)
        initial_s[np.ravel_multi_index((3,0), self.shape)] = 1.0
        super(WindyGridWorld, self).__init__(nS, nA, P, initial_s)

    def _calculate_transition_prob(self, current, move, winds):
        # Transition probability for a position landed on is 1.0, new_state calculated
        new_position = np.array(current) + np.array(move) + np.array([-1, 0]) * winds[(tuple(current))]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.goal
        return [(1.0, new_state, -1, is_done)]

    def _limit_coordinates(self, coord):
        for i in range(2):
            coord[i] = min(coord[i], self.shape[i] - 1)
            coord[i] = max(0, coord[i])
        return coord

    def render(self):
        outfile = sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif position == self.goal:
                output = " T "
            else:
                output = " o "
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"
            outfile.write(output)
        outfile.write("\n")
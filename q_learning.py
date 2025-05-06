"""
Snake Eater Q learning basic algorithm
Made with PyGame
Machine Learning Classes - University Carlos III of Madrid
"""

import numpy as np
import random

class QLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_min=0.01, epsilon_decay=0.999999):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.load_q_table()

        # params to decrease learning rate over time
        self.alpha_decay=0.99999
        self.alpha_min = 0.01

    def choose_action(self, state, allowed_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(allowed_actions)  # Explore
        else:
            q_vals = self.q_table[state, allowed_actions] # Exploit from allowed actions
            best = np.argmax(q_vals)
            action = allowed_actions[best]
            
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        # Your code here
        # Update the current Q-value using the Q-learning formula
        # if terminal_state:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        # else:
        # Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))
        
        if next_state is None:
            # Terminal state: Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
            self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + self.alpha * reward
        else:
            # Non-terminal: Include discounted max Q(nextState)
            max_next_q = max(self.q_table[next_state][action]) 
            self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + self.alpha * (reward + self.discount * max_next_q)
        """
        # Your code here
        reward = np.clip(reward, -10, 20)  # Prevent extreme rewards
        
        current_q = self.q_table[state, action]

        # Terminal state handling
        if next_state is None:
            # No future rewards from terminal state
            new_q = (1 - self.alpha) * current_q + self.alpha * (reward + 0) 
            #current_q + self.alpha * (reward - current_q)
        else:
            # Normal Bellman equation update
            max_next_q = np.max(self.q_table[next_state])
            new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)

        self.q_table[state, action] = new_q

        # Decay alpha
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
    def save_q_table(self, filename="q_table.txt"):
        np.savetxt(filename, self.q_table)

    def load_q_table(self, filename="q_table.txt"):
        try:
            self.q_table = np.loadtxt(filename)
        except IOError:
            self.q_table = np.zeros((self.n_states, self.n_actions))



"""
Snake Eater Environment
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""

import numpy as np
import random
import math

class SnakeGameEnv:
    def __init__(self, frame_size_x=150, frame_size_y=150, growing_body=True):
        # Initializes the environment with default values
        self.frame_size_x = frame_size_x
        self.frame_size_y = frame_size_y
        self.growing_body = growing_body
        self.reset()

    def reset(self):
        # Resets the environment with default values
        self.snake_pos = [50, 50]
        self.snake_body = [[50, 50], [60, 50], [70, 50]]
        self.food_pos = [random.randrange(1, (self.frame_size_x // 10)) * 10, random.randrange(1, (self.frame_size_y // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.game_over = False
        return self.get_state()

    def step(self, action):
        # Implements the logic to change the snake's direction based on action
        # Update the snake's head position based on the direction
        # Check for collision with food, walls, or self
        # Update the score and reset food as necessary
        # Determine if the game is over
        self.update_snake_position(action)
        reward = self.calculate_reward()
        self.update_food_position()
        state = self.get_state()
        self.game_over = self.check_game_over()
        return state, reward, self.game_over

    def get_state(self):
        # Your code here: return the state of the environment
        # now returns 15 features: 
        # 0–9: as before, 10 new: wall distances (4) + nearest body distance (1)
        state = np.zeros(15)
        
        head_x = self.snake_pos[0]
        head_y = self.snake_pos[1]
        dx = self.food_pos[0] - head_x
        dy = self.food_pos[1] - head_y
        
        # Danger detection (straight, right, left)
        danger_straight, danger_right, danger_left = False, False, False
        if self.direction == 'UP':
            danger_straight = (head_y - 10 < 0) or ([head_x, head_y - 10] in self.snake_body[1:])
            danger_right    = (head_x + 10 >= self.frame_size_x) or ([head_x + 10, head_y] in self.snake_body[1:])
            danger_left     = (head_x - 10 < 0) or ([head_x - 10, head_y] in self.snake_body[1:])
            food_forward    = dy < 0
            food_right      = dx > 0
        elif self.direction == 'DOWN':
            danger_straight = (head_y + 10 >= self.frame_size_y) or ([head_x, head_y + 10] in self.snake_body[1:])
            danger_right    = (head_x - 10 < 0) or ([head_x - 10, head_y] in self.snake_body[1:])
            danger_left     = (head_x + 10 >= self.frame_size_x) or ([head_x + 10, head_y] in self.snake_body[1:])
            food_forward    = dy > 0
            food_right      = dx < 0
        elif self.direction == 'LEFT':
            danger_straight = (head_x - 10 < 0) or ([head_x - 10, head_y] in self.snake_body[1:])
            danger_right    = (head_y + 10 >= self.frame_size_y) or ([head_x, head_y + 10] in self.snake_body[1:])
            danger_left     = (head_y - 10 < 0) or ([head_x, head_y - 10] in self.snake_body[1:])
            food_forward    = dx < 0
            food_right      = dy < 0
        else:  # RIGHT
            danger_straight = (head_x + 10 >= self.frame_size_x) or ([head_x + 10, head_y] in self.snake_body[1:])
            danger_right    = (head_y - 10 < 0) or ([head_x, head_y - 10] in self.snake_body[1:])
            danger_left     = (head_y + 10 >= self.frame_size_y) or ([head_x, head_y + 10] in self.snake_body[1:])
            food_forward    = dx > 0
            food_right      = dy > 0
        
        # original 10 features
        state[0] = head_x / self.frame_size_x
        state[1] = head_y / self.frame_size_y
        state[2] = dx / self.frame_size_x
        state[3] = dy / self.frame_size_y
        state[4] = int(danger_straight)
        state[5] = int(danger_right)
        state[6] = int(danger_left)
        state[7] = int(food_forward)
        state[8] = int(food_right)
        state[9] = len(self.snake_body) / 20
        
        max_steps_x = self.frame_size_x // 10
        max_steps_y = self.frame_size_y // 10
        
        dist_up    = head_y // 10
        dist_down  = (self.frame_size_y - head_y - 10) // 10
        dist_left  = head_x // 10
        dist_right = (self.frame_size_x - head_x - 10) // 10
        
        state[10] = dist_up    / max_steps_y
        state[11] = dist_down  / max_steps_y
        state[12] = dist_left  / max_steps_x
        state[13] = dist_right / max_steps_x
        
        if len(self.snake_body) > 1:
            dists = [math.hypot(seg[0]-head_x, seg[1]-head_y) for seg in self.snake_body[1:]]
            min_body = min(dists)
        else:
            min_body = math.hypot(self.frame_size_x, self.frame_size_y)
        # normalize by board diagonal
        state[14] = min_body / math.hypot(self.frame_size_x, self.frame_size_y)
        
        return state

    def get_body(self):
        return self.snake_body

    def get_food(self):
        return self.food_pos

    def check_game_over(self):
        # Return True if the game is over, else False
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            print("Snake hit x wall")
            return True
        if self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:
            print("Snake hit y wall")
            return True
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                print("Snake hit itself")
                return True
                
        return False

    def calculate_reward(self):
        """
        Simplified reward function with better incentives for survival and food-seeking
        """
        # base case - small positive reward for survival
        reward = 0.1
        
        # terminal case - game over
        if self.check_game_over():
            return -10.0  # significant but not extreme penalty
        
        # food reward
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            print("Snake ate food")
            return 12.0  # good but not overwhelming reward
        
        # distance-based reward shaping - use manhattan distance for simplicity
        curr_dist = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
        prev_head = self.snake_body[1] if len(self.snake_body) > 1 else self.snake_pos
        prev_dist = abs(prev_head[0] - self.food_pos[0]) + abs(prev_head[1] - self.food_pos[1])
        
        # reward for getting closer to food (normalized by board size)
        max_dist = self.frame_size_x + self.frame_size_y
        delta = (prev_dist - curr_dist) / max_dist
        reward += delta * 5.0  # scale the distance reward
        
        # add slight penalty for very long games to encourage food-seeking
        reward -= 0.001 * (len(self.snake_body) - 3)  # small penalty based on steps taken
        
        return reward


    def update_snake_position(self, action):
        # Updates the snake's position based on the action
        # Map action to direction
        change_to = ''
        direction = self.direction
        if action == 0:
            change_to = 'UP'
        elif action == 1:
            change_to = 'DOWN'
        elif action == 2:
            change_to = 'LEFT'
        elif action == 3:
            change_to = 'RIGHT'
    
        # Move the snake
        if change_to == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if change_to == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if change_to == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if change_to == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'
    
        if direction == 'UP':
            self.snake_pos[1] -= 10
        elif direction == 'DOWN':
            self.snake_pos[1] += 10
        elif direction == 'LEFT':
            self.snake_pos[0] -= 10
        elif direction == 'RIGHT':
            self.snake_pos[0] += 10
            
        self.direction = direction
        
        
        self.snake_body.insert(0, list(self.snake_pos))
        
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 10
            self.food_spawn = False
            # If the snake is not growing
            if not self.growing_body:
                self.snake_body.pop()
        else:
            self.snake_body.pop()
    
    def update_food_position(self):
        if not self.food_spawn:
            self.food_pos = [random.randrange(1, (self.frame_size_x//10)) * 10, random.randrange(1, (self.frame_size_x//10)) * 10]
        self.food_spawn = True

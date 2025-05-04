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
        # Your code here
        # 7 main attributes: head, tail, distance to food, danger of walls. generic but meaningful to reward and bellman
        state = np.zeros(8)  
        
        head_x = self.snake_pos[0]  # head x
        head_y = self.snake_pos[1]  # head y
        dx = self.snake_pos[0] - self.food_pos[0]  # x distance to food
        dy = self.snake_pos[1] - self.food_pos[1]  # y distance to food
        
        # Add danger detection in three directions (relative to current direction)
        danger_straight, danger_right, danger_left, danger_behind = False, False, False, False
        
        # Check for danger based on current direction
        if self.direction == 'UP':
            # Danger straight = wall above or body part above
            danger_straight = (head_y - 10 < 0) or ([head_x, head_y - 10] in self.snake_body[1:])
            # Danger right = wall to the right or body part to the right
            danger_right = (head_x + 10 >= self.frame_size_x) or ([head_x + 10, head_y] in self.snake_body[1:])
            # Danger left = wall to the left or body part to the left
            danger_left = (head_x - 10 < 0) or ([head_x - 10, head_y] in self.snake_body[1:])

            danger_behind = (head_y + 10 >= self.frame_size_y) or ([head_x, head_y + 10] in self.snake_body[1:])
        elif self.direction == 'DOWN':
            danger_straight = (head_y + 10 >= self.frame_size_y) or ([head_x, head_y + 10] in self.snake_body[1:])
            danger_right = (head_x - 10 < 0) or ([head_x - 10, head_y] in self.snake_body[1:])
            danger_left = (head_x + 10 >= self.frame_size_x) or ([head_x + 10, head_y] in self.snake_body[1:])
            danger_behind = (head_y - 10 < 0) or ([head_x, head_y - 10] in self.snake_body[1:])
        elif self.direction == 'LEFT':
            danger_straight = (head_x - 10 < 0) or ([head_x - 10, head_y] in self.snake_body[1:])
            danger_right = (head_y + 10 >= self.frame_size_y) or ([head_x, head_y + 10] in self.snake_body[1:])
            danger_left = (head_y - 10 < 0) or ([head_x, head_y - 10] in self.snake_body[1:])
            danger_behind = (head_x + 10 >= self.frame_size_x) or ([head_x + 10, head_y] in self.snake_body[1:])
        elif self.direction == 'RIGHT':
            danger_straight = (head_x + 10 >= self.frame_size_x) or ([head_x + 10, head_y] in self.snake_body[1:])
            danger_right = (head_y - 10 < 0) or ([head_x, head_y - 10] in self.snake_body[1:])
            danger_left = (head_y + 10 >= self.frame_size_y) or ([head_x, head_y + 10] in self.snake_body[1:])
            danger_behind = (head_x - 10 < 0) or ([head_x - 10, head_y] in self.snake_body[1:])

        state[0] = head_x
        state[1] = head_y
        state[2] = dx
        state[3] = dy
        state[4] = int(danger_straight)  # Convert boolean to int (0 or 1)
        state[5] = int(danger_right)
        state[6] = int(danger_left)
        state[7] = int(danger_behind)
        
        return state

    def get_body(self):
        return self.snake_body

    def get_food(self):
        return self.food_pos

    # def calculate_reward(self):
    #     """
    #     # Your code here
    #     # Calculate and return the reward. Remember that you can provide possitive or negative reward.
    #     head, tail, head_to_food, tail_to_food = self.get_state() 
    #     reward = 1 + (-.25 * head_to_food) + (-.25 * tail_to_food)
    #     return reward
    #     """

    #     head_x, head_y, x_to_food, y_to_food, _, _, _, _ = self.get_state()
    #     food_x, food_y = self.food_pos
        
    #     # Check if food eaten
    #     if head_x == food_x and head_y == food_y:
    #         return 15
        
    #     # big penalty if die
    #     if self.check_game_over():
    #         return -20
        
    #     # Small penalty for moving into dangerous positions. learn to avoid dangerous situations even before dying
    #     # if danger_straight or danger_right or danger_left:
    #     #     return -0.1
        
    #     # # Encourage moving towards food
    #     # # prev and curr dist to food
    #     # old_head_pos = self.snake_body[1]  # Tail is Previous head position!!

    #     # # manhattan dist
    #     # old_distance = abs(old_head_pos[0] - self.food_pos[0]) + abs(old_head_pos[1] - self.food_pos[1])
    #     # new_distance = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
        
    #     # # Small reward for moving toward food, small penalty for moving away
    #     # if new_distance < old_distance:
    #     #     return 0.1
    #     # else:
    #     #     return -0.05
    #     new_dist = abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
    #     old_head = self.snake_body[1] if len(self.snake_body) > 1 else self.snake_pos
    #     old_dist = abs(old_head[0] - self.food_pos[0]) + abs(old_head[1] - self.food_pos[1])
        
    #     # Progressive rewards and penalties
    #     reward = 0
    #     reward += 0.5 if new_dist < old_dist else -0.3
    #     reward -= 0.05  # Time penalty per step
    #     reward -= 0.1 * sum(self.get_state()[4:7])  # FIXME Danger penalty
    #     return reward

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
        # Your code here
        """
        Calculate reward based on game events and movement efficiency.
        """
        # Food-related rewards
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            return 10.0  # Major reward for eating food
            
        # Death penalty
        if self.check_game_over():
            return -10.0  # High penalty for dying
        
        # Movement rewards/penalties
        head_x, head_y = self.snake_pos
        food_x, food_y = self.food_pos
        
        # Calculate distances
        new_dist = abs(head_x - food_x) + abs(head_y - food_y)
        old_head = self.snake_body[1] if len(self.snake_body) > 1 else self.snake_pos
        old_dist = abs(old_head[0] - food_x) + abs(old_head[1] - food_y)
        
        # Reward moving toward food, penalize moving away
        if new_dist < old_dist:
            return 0.1  # Small reward for getting closer to food
        elif new_dist > old_dist:
            return -0.1  # Small penalty for moving away from food
        
        # Small penalty for each step to discourage loops/waiting
        return -0.01

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

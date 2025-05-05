"""
Snake Eater Game
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""

import numpy as np
from snake_env import SnakeGameEnv
from q_learning import QLearning
import pygame
import sys
import math

def get_state_index(state, env):
    """Convert continuous state to discrete index using new state features"""
    # Extract normalized values from state
    head_x_norm, head_y_norm, dx_norm, dy_norm, danger_straight, danger_right, danger_left, food_forward, food_right, body_length = state
    
    # Convert normalized distances back to pixels for binning
    dx = dx_norm * env.frame_size_x
    dy = dy_norm * env.frame_size_y
    
    # Food direction (8 possibilities - combines absolute and relative)
    if food_forward:
        direction = 0 if food_right else 1  # 0: front-right, 1: front-left
    else:
        if abs(dx) > abs(dy):  # Horizontal alignment stronger
            direction = 2 if dx > 0 else 3  # 2: right, 3: left
        else:  # Vertical alignment stronger
            direction = 4 if dy > 0 else 5  # 4: down, 5: up
    
    # Distance bins (4 possibilities)
    distance = math.sqrt(dx**2 + dy**2)
    distance_bin = min(3, int(distance // (env.frame_size_x/4)))  # 4 bins
    
    # Danger level (3 possibilities: 0, 1, or 2+ dangers)
    danger_level = min(2, danger_straight + danger_right + danger_left)
    
    # Body length bins (3 possibilities)
    length_bin = min(2, int(body_length * 20 // 7))  # 3 bins
    
    # Combined index (6 dir × 4 dist × 3 danger × 3 length = 216 total)
    return int((direction * 36) + (distance_bin * 9) + (danger_level * 3) + length_bin)

def get_safe_actions(env, lookahead_steps=2):
    """Updated to use normalized positions and new danger detection"""
    safe_actions = []
    current_head = env.snake_pos.copy()
    current_body = env.snake_body.copy()
    state = env.get_state()  # Use new state representation
    
    for action in [0, 1, 2, 3]:
        # Skip 180-degree reversals
        if (env.direction == 'UP' and action == 1) or \
           (env.direction == 'DOWN' and action == 0) or \
           (env.direction == 'LEFT' and action == 3) or \
           (env.direction == 'RIGHT' and action == 2):
            continue
            
        # Check immediate danger using state flags
        if action == 0:  # UP
            if state[4]: continue  # danger_straight when facing up
        elif action == 1:  # DOWN
            if state[4] and env.direction == 'DOWN': continue
        elif action == 2:  # LEFT
            if state[6]: continue  # danger_left
        elif action == 3:  # RIGHT
            if state[5]: continue  # danger_right
            
        # Advanced collision prediction
        virtual_head = current_head.copy()
        if action == 0: virtual_head[1] -= 10
        elif action == 1: virtual_head[1] += 10
        elif action == 2: virtual_head[0] -= 10
        elif action == 3: virtual_head[0] += 10
        
        # Normalized position check
        if (virtual_head[0] < 0 or virtual_head[0] >= env.frame_size_x or 
            virtual_head[1] < 0 or virtual_head[1] >= env.frame_size_y):
            continue
            
        # Body collision check using new state info
        if state[9] > 0.15:  # Only check if snake has some length
            future_body = [virtual_head] + current_body[:-1]
            if any(pos == virtual_head for pos in future_body[:int(state[9]*20)]):
                continue
                
        safe_actions.append(action)
    
    return safe_actions if safe_actions else [0, 1, 2, 3]

def main():
    # Window size
    FRAME_SIZE_X = 150
    FRAME_SIZE_Y = 150
    
    # Colors (R, G, B)
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    GREEN = pygame.Color(0, 255, 0)
    BLUE = pygame.Color(0, 0, 255)
    
    difficulty = 20  # Adjust as needed
    render_game = True # Show the game or not
    growing_body = True # Makes the body of the snake grow
    training = True # Defines if it should train or not

    # Initialize the game window, environment and q_learning algorithm
    # Your code here.
    # You must define the number of possible states.

    number_states = 288 # 80

    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)
    ql = QLearning(
        n_states=number_states, 
        n_actions=4,
        alpha=0.1,       # Higher learning rate
        gamma=0.98,      # Slightly more future-focused
        epsilon=1.0,
        epsilon_min=0.1,  # Maintain some exploration
        epsilon_decay=0.995  # Slower decay
    )

    num_episodes = 1000 # the number of episodes you want for training.
        
    
    if render_game:
        game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        fps_controller = pygame.time.Clock()
    
    avg_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        game_over = False
        steps = 0

        # Maximum steps per episode to prevent infinite loops and bad training
        max_steps = min(
            400 + (episode // 8),  # Gradually increase from 400 to 500 steps
            500 
        ) if training else 1000

        while not game_over and steps < max_steps:
            steps += 1

            # discretize state: distance to food
            state_index = get_state_index(state, env)
            
            if training and episode < 150:
                # During early training, use all actions (no lookahead)
                allowed_actions = [0, 1, 2, 3]
                # Exclude only direct reversals
                if env.direction == 'UP': 
                    allowed_actions.remove(1)  # remove DOWN
                elif env.direction == 'DOWN': 
                    allowed_actions.remove(0)  # remove UP
                elif env.direction == 'LEFT': 
                    allowed_actions.remove(3)  # remove RIGHT
                elif env.direction == 'RIGHT': 
                    allowed_actions.remove(2)  # remove LEFT
            else:
                # Later in training and during evaluation, use safety lookahead
                lookahead = 2
                allowed_actions = get_safe_actions(env, lookahead)

            # Choose action using epsilon-greedy strategy
            action = ql.choose_action(state_index, allowed_actions)

            # Step environment
            next_state, reward, game_over = env.step(action)

            # Get index for next state
            if not game_over:
                next_state_index = get_state_index(next_state, env)
            else:
                next_state_index = None

            # Training step: update Q-table
            if training:
                ql.update_q_table(state_index, action, reward, next_state_index)

            # Update state and reward
            state = next_state
            total_reward += reward

            # Render
            if render_game:
                game_window.fill(BLACK)
                snake_body = env.get_body()
                food_pos = env.get_food()
                for pos in snake_body:
                    pygame.draw.rect(game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))
        
                pygame.draw.rect(game_window, RED, pygame.Rect(food_pos[0], food_pos[1], 10, 10))
            
            if env.check_game_over():
                break
                
            if render_game:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    
                pygame.display.flip()
                fps_controller.tick(difficulty)
        
        print(f"Episode {episode+1}, Total reward: {total_reward}, Epsilon: {ql.epsilon}, Steps: {steps}")
        ql.save_q_table()
        avg_rewards.append(total_reward)
    
    print(f"Average reward: {sum(avg_rewards[950:])/len(avg_rewards[950:])}")
    
if __name__ == "__main__":
    main()







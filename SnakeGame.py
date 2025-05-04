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

def get_state_index(state):
    # Your code here
    """
    Convert continuous state to discrete index for Q-table.
    Total states: 4 × 5 × 2 = 40 states
    """
    _, _, dx, dy, danger_straight, danger_right, danger_left, _ = state
    
    # Food direction relative to snake (4 possibilities)
    if abs(dx) > abs(dy):  # Horizontal alignment is stronger
        direction = 0 if dx > 0 else 1  # 0: food is right, 1: food is left
    else:  # Vertical alignment is stronger
        direction = 2 if dy > 0 else 3  # 2: food is down, 3: food is up
    
    # Simplified distance (5 bins)
    distance = abs(dx) + abs(dy)
    distance_bin = min(4, distance // 30)  # 5 distance bins (0-4)
    
    # Simplified danger (2 states)
    # Just check if any immediate danger exists
    danger = 1 if (danger_straight or danger_right or danger_left) else 0
    
    # Calculate state index: direction × 5 × 2 + distance_bin × 2 + danger
    # This gives 40 unique states (4 directions × 5 distances × 2 danger states)
    return int(direction * 10 + distance_bin * 2 + danger)

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
    training = False # Defines if it should train or not

    # Initialize the game window, environment and q_learning algorithm
    # Your code here.
    # You must define the number of possible states.

    number_states = 40 # 80

    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)
    ql = QLearning(
        n_states=number_states, 
        n_actions=4,
        alpha=0.05,      
        gamma=0.95,    
        epsilon=1.0,    # Start fully exploratory
        epsilon_min=0.05,  
        epsilon_decay=0.999  # Slow decay
    )

    num_episodes = 1000 # the number of episodes you want for training.

    def get_safe_actions(env, lookahead_steps=2):
        # your code here
        """returns only actions that won't cause immediate or predictable collisions
        args:
            env: the snake game environment
            lookahead_steps: how many future moves to simulate 
        returns:
            list of safe action indices (0=up, 1=down, 2=left, 3=right)
        """
        safe_actions = []
        current_head = env.snake_pos.copy()
        current_body = env.snake_body.copy()
        
        for action in [0, 1, 2, 3]:
            # skip if this is a direct 180-degree reversal
            if (env.direction == 'UP' and action == 1) or \
            (env.direction == 'DOWN' and action == 0) or \
            (env.direction == 'LEFT' and action == 3) or \
            (env.direction == 'RIGHT' and action == 2):
                continue
                
            # simulate movement for this action
            virtual_head = current_head.copy()
            if action == 0: virtual_head[1] -= 10  # up
            elif action == 1: virtual_head[1] += 10  # down
            elif action == 2: virtual_head[0] -= 10  # left
            elif action == 3: virtual_head[0] += 10  # right
            
            # check immediate wall collision
            if (virtual_head[0] < 0 or virtual_head[0] >= env.frame_size_x or 
                virtual_head[1] < 0 or virtual_head[1] >= env.frame_size_y):
                continue
                
            # check immediate body collision (excluding tail which will move)
            if virtual_head in current_body[:-1]:
                continue
                
            # if growing, check future body positions
            if env.growing_body and lookahead_steps > 0:
                collision_found = False
                future_body = [virtual_head] + current_body[:-1]
                
                # simulate next moves assuming same direction (conservative estimate)
                for _ in range(lookahead_steps):
                    next_head = future_body[0].copy()
                    if action == 0: next_head[1] -= 10
                    elif action == 1: next_head[1] += 10
                    elif action == 2: next_head[0] -= 10
                    elif action == 3: next_head[0] += 10
                    
                    # check future collisions
                    if (next_head in future_body or 
                        next_head[0] < 0 or next_head[0] >= env.frame_size_x or
                        next_head[1] < 0 or next_head[1] >= env.frame_size_y):
                        collision_found = True
                        break
                        
                    future_body.insert(0, next_head)
                    if not env.growing_body:
                        future_body.pop()
                
                if collision_found:
                    continue
                    
            # if we get here, action is safe
            safe_actions.append(action)
        
        # fallback if all actions are dangerous (should be rare)
        return safe_actions if safe_actions else [0, 1, 2, 3]
        
    
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
            state_index = get_state_index(state)
            
            if training and episode < 300:
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
                lookahead = 1 if episode < 500 else 2
                allowed_actions = get_safe_actions(env, lookahead)
            allowed_actions = get_safe_actions(env, lookahead)  # all actions initially allowed

            # Choose action using epsilon-greedy strategy
            action = ql.choose_action(state_index, allowed_actions)

            # Step environment
            next_state, reward, game_over = env.step(action)

            # Get index for next state
            if not game_over:
                next_state_index = get_state_index(next_state)
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






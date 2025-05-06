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
    """Convert continuous state to discrete index using enhanced state features"""
    # Your code here
    head_x_norm, head_y_norm, dx_norm, dy_norm, danger_straight, danger_right, danger_left, food_forward, food_right, body_length, wall_up, wall_down, wall_left, wall_right, body_dist = state
    
    # Convert normalized distances back to pixels for binning
    dx = dx_norm * env.frame_size_x
    dy = dy_norm * env.frame_size_y
    
    # Food direction (8 possibilities - combines absolute and relative)
    if food_forward:
        direction = 0 if food_right else 1  # 0: front-right, 1: front-left
    else:
        if abs(dx) > abs(dy):  
            direction = 2 if dx > 0 else 3  # 2: right, 3: left
        else:  
            direction = 4 if dy > 0 else 5  # 4: down, 5: up
    
    distance = math.sqrt(dx**2 + dy**2)
    distance_bin = min(3, int(distance // (env.frame_size_x/4)))  # 4 bins
    
    danger_level = min(2, danger_straight + danger_right + danger_left)
    
    # Wall proximity (2 possibilities: close to wall or not)
    wall_proximity = 0
    if min(wall_up, wall_down, wall_left, wall_right) < 0.2:  # Close to at least one wall
        wall_proximity = 1
    
    # Body proximity (2 possibilities: close to body or not)
    body_proximity = 0
    if body_dist < 0.2 and body_length > 0.2:  # Close to body and has some length
        body_proximity = 1
    
    
    # Combined index (6 dir × 4 dist × 3 danger × 2 wall × 2 body × 3 length = 864 total)
    return int((direction * 48) + (distance_bin * 12) + (danger_level * 4) + (wall_proximity * 2) + body_proximity)

def get_safe_actions(env, lookahead_steps=2):
    """Returns actions that won't lead to collisions, with special handling for edge food"""
    safe_actions = []
    food_actions = []  # track actions that lead directly to food
    current_head = env.snake_pos.copy()
    current_body = env.snake_body.copy()
    current_direction = env.direction
    food_pos = env.food_pos
    
    # Map directions to actions and back
    direction_to_action = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
    action_to_direction = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    incompatible = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
    
    # Check if food is at the edge
    food_at_edge = (food_pos[0] == 0 or food_pos[0] == env.frame_size_x - 10 or 
                    food_pos[1] == 0 or food_pos[1] == env.frame_size_y - 10)
    
    for action in [0, 1, 2, 3]:
        new_direction = action_to_direction[action]
        
        # Skip 180-degree reversals
        if new_direction == incompatible.get(current_direction):
            continue
            
        # Calculate next position
        next_head = current_head.copy()
        if new_direction == 'UP':
            next_head[1] -= 10
        elif new_direction == 'DOWN':
            next_head[1] += 10
        elif new_direction == 'LEFT':
            next_head[0] -= 10
        elif new_direction == 'RIGHT':
            next_head[0] += 10
        
        # Check if this action leads to food
        leads_to_food = (next_head[0] == food_pos[0] and next_head[1] == food_pos[1])
        if leads_to_food:
            food_actions.append(action)
            
        # Simulate the move for safety
        is_safe = True
        virtual_head = current_head.copy()
        virtual_body = current_body.copy()
        virtual_direction = current_direction
        
        # Perform lookahead
        for step in range(lookahead_steps):
            # Update direction based on action 
            if step == 0:
                virtual_direction = new_direction
            
            # Move virtual head according to direction
            if virtual_direction == 'UP':
                virtual_head[1] -= 10
            elif virtual_direction == 'DOWN':
                virtual_head[1] += 10
            elif virtual_direction == 'LEFT':
                virtual_head[0] -= 10
            elif virtual_direction == 'RIGHT':
                virtual_head[0] += 10
            
            # Check wall collision
            if (virtual_head[0] < 0 or virtual_head[0] >= env.frame_size_x or 
                virtual_head[1] < 0 or virtual_head[1] >= env.frame_size_y):
                is_safe = False
                break
                
            if step > 0 or not (leads_to_food and food_at_edge):
                # Check body collision 
                check_body = virtual_body[:-1] if step == 0 else virtual_body
                if any(virtual_head[0] == segment[0] and virtual_head[1] == segment[1] for segment in check_body):
                    is_safe = False
                    break
            
            # Update virtual body for next step simulation
            virtual_body.insert(0, virtual_head.copy())
            virtual_body.pop()
        
        if is_safe:
            safe_actions.append(action)
    
    if food_actions and food_at_edge:
        # Find actions that both lead to food and are considered safe
        safe_food_actions = [a for a in food_actions if a in safe_actions]
        if safe_food_actions:
            print("Edge food actions:", safe_food_actions)
            return safe_food_actions
        elif len(env.snake_body) > 5:  # Only take risks when snake has some length
            print("Taking risk for edge food:", food_actions)
            return food_actions
            
    print("Safe actions:", safe_actions)
    
    # If no safe actions, return actions that avoid immediate wall collisions
    if not safe_actions:
        immediate_safe = []
        for action in [0, 1, 2, 3]:
            new_direction = action_to_direction[action]
            
            if new_direction == incompatible.get(current_direction):
                continue
                
            # Check only immediate wall collision
            test_head = current_head.copy()
            if new_direction == 'UP':
                test_head[1] -= 10
            elif new_direction == 'DOWN':
                test_head[1] += 10
            elif new_direction == 'LEFT':
                test_head[0] -= 10
            elif new_direction == 'RIGHT':
                test_head[0] += 10
                
            if (test_head[0] >= 0 and test_head[0] < env.frame_size_x and 
                test_head[1] >= 0 and test_head[1] < env.frame_size_y):
                immediate_safe.append(action)
        
        print("No safe actions. Using wall-avoiding actions:", immediate_safe)
        return immediate_safe if immediate_safe else [0, 1, 2, 3]
    
    return safe_actions

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
    
    difficulty = 25  # Adjust as needed
    render_game = True # Show the game or not
    growing_body = True # Makes the body of the snake grow
    training = False # Defines if it should train or not

    # Initialize the game window, environment and q_learning algorithm
    # Your code here.
    # You must define the number of possible states.

    number_states = 288 # 80

    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)
    ql = QLearning(
        n_states=number_states, 
        n_actions=4,
        alpha=0.2,       # higher learning rate to learn faster
        gamma=0.95,      # balance between immediate and future rewards
        epsilon=1.0,     # start fully exploratory
        epsilon_min=0.1,  # maintain some exploration
        epsilon_decay=0.999  # slower decay for more exploration
    )

    num_episodes = 5000  # the number of episodes you want for training.
        
    
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









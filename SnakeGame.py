"""
Snake Eater Game
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""

from snake_env import SnakeGameEnv
from q_learning import QLearning
import pygame
import sys


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

    number_states = 80

    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)
    ql = QLearning(n_states=number_states, n_actions=4)  

    num_episodes = 500 # the number of episodes you want for training.


    if render_game:
        game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        fps_controller = pygame.time.Clock()
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        game_over = False

        while not game_over:
            # Discretize the state: direction of food and distance
            dx = state[2]
            dy = state[3]

            # Get direction of food relative to head (8 directions)
            if dx > 0 and dy == 0:
                direction = 0  # right
            elif dx < 0 and dy == 0:
                direction = 1  # left
            elif dx == 0 and dy > 0:
                direction = 2  # down
            elif dx == 0 and dy < 0:
                direction = 3  # up
            elif dx > 0 and dy > 0:
                direction = 4  # down-right
            elif dx > 0 and dy < 0:
                direction = 5  # up-right
            elif dx < 0 and dy > 0:
                direction = 6  # down-left
            elif dx < 0 and dy < 0:
                direction = 7  # up-left

            # Get approximate distance bin (1–10)
            distance = min(10, int((abs(dx) + abs(dy)) / 10) + 1)

            # Construct state index: direction * 10 + (distance - 1)
            state_index = direction * 10 + (distance - 1)
            allowed_actions = []
            if state[0] != FRAME_SIZE_X:
                allowed_actions.append(3)  # RIGHT (3)
            if state[0] != 0:
                allowed_actions.append(2)  # LEFT (2)
            if state[1] != FRAME_SIZE_Y:
                allowed_actions.append(1)  # DOWN (1)
            if state[1] != 0:
                allowed_actions.append(0)  # UP (0)

            # Choose action using epsilon-greedy strategy
            action = ql.choose_action(state_index, allowed_actions)

            # Step environment
            next_state, reward, game_over = env.step(action)

            # Discretize next state
            dx_n = next_state[2]
            dy_n = next_state[3]
            if dx_n > 0 and dy_n == 0:
                direction_n = 0
            elif dx_n < 0 and dy_n == 0:
                direction_n = 1
            elif dx_n == 0 and dy_n > 0:
                direction_n = 2
            elif dx_n == 0 and dy_n < 0:
                direction_n = 3
            elif dx_n > 0 and dy_n > 0:
                direction_n = 4
            elif dx_n > 0 and dy_n < 0:
                direction_n = 5
            elif dx_n < 0 and dy_n > 0:
                direction_n = 6
            elif dx_n < 0 and dy_n < 0:
                direction_n = 7

            distance_n = min(10, int((abs(dx_n) + abs(dy_n)) / 10) + 1)
            next_state_index = direction_n * 10 + (distance_n - 1)

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
        
        ql.save_q_table()
        print(f"Episode {episode+1}, Total reward: {total_reward}")

    
if __name__ == "__main__":
    main()



"""
Snake Eater Game
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
"""
Snake Eater Game
Made with PyGame
Last modification in April 2024 by José Luis Perán
Machine Learning Classes - University Carlos III of Madrid
"""
"""
from snake_env import SnakeGameEnv
from q_learning import QLearning
import pygame
import sys
import numpy as np

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
    
    difficulty = 5  # Adjust as needed
    render_game = True # Show the game or not
    growing_body = True # Makes the body of the snake grow
    training = True # Defines if it should train or not

    # Initialize the game window, environment and q_learning algorithm
    pygame.init()
    env = SnakeGameEnv(FRAME_SIZE_X, FRAME_SIZE_Y, growing_body)
    
    # Define state space size based on discretized grid
    grid_size = 10
    #x_states = FRAME_SIZE_X // grid_size  # 15 possible x positions
    #y_states = FRAME_SIZE_Y // grid_size  # 15 possible y positions
    #food_directions = 8  # 8 possible relative food positions (N, NE, E, SE, S, SW, W, NW)
    #number_states = x_states * y_states * food_directions
    number_states = 4
    
    ql = QLearning(n_states=number_states, n_actions=4)

    if render_game:
        game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
        fps_controller = pygame.time.Clock()
    
    num_episodes = 1000  # Increased number of episodes for better training
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        game_over = False
        
        while not game_over:
            # Get current state information
            head_x, head_y = env.snake_pos
            food_x, food_y = env.food_pos
            
            # Discretize head position
            head_x_discrete = head_x // grid_size
            head_y_discrete = head_y // grid_size
            
            # Calculate relative food position (8 directions)
            dx = food_x - head_x
            dy = food_y - head_y
            
            # Determine food direction (simplified to 8 directions)
            if dx == 0 and dy < 0:
                food_dir = 0  # N
            elif dx > 0 and dy < 0:
                food_dir = 1  # NE
            elif dx > 0 and dy == 0:
                food_dir = 2  # E
            elif dx > 0 and dy > 0:
                food_dir = 3  # SE
            elif dx == 0 and dy > 0:
                food_dir = 4  # S
            elif dx < 0 and dy > 0:
                food_dir = 5  # SW
            elif dx < 0 and dy == 0:
                food_dir = 6  # W
            elif dx < 0 and dy < 0:
                food_dir = 7  # NW
            else:
                food_dir = 0  # Default
            
            # Calculate state index
            state_idx = (head_x_discrete * y_states + head_y_discrete) * food_directions + food_dir
            
            # Ensure state index is within bounds
            state_idx = min(state_idx, number_states - 1)
            
            # Get possible actions (considering walls)
            possible_actions = []
            if head_x < FRAME_SIZE_X - 10:
                possible_actions.append(3)  # RIGHT
            if head_x > 0:
                possible_actions.append(2)  # LEFT
            if head_y < FRAME_SIZE_Y - 10:
                possible_actions.append(1)  # DOWN
            if head_y > 0:
                possible_actions.append(0)  # UP
                
            # Choose action
            action = ql.choose_action(state_idx, possible_actions)
            
            # Execute action
            next_state, reward, game_over = env.step(action)
            
            # Calculate next state index
            next_head_x, next_head_y = env.snake_pos
            next_food_x, next_food_y = env.food_pos
            
            next_head_x_discrete = next_head_x // grid_size
            next_head_y_discrete = next_head_y // grid_size
            
            next_dx = next_food_x - next_head_x
            next_dy = next_food_y - next_head_y
            
            if next_dx == 0 and next_dy < 0:
                next_food_dir = 0  # N
            elif next_dx > 0 and next_dy < 0:
                next_food_dir = 1  # NE
            elif next_dx > 0 and next_dy == 0:
                next_food_dir = 2  # E
            elif next_dx > 0 and next_dy > 0:
                next_food_dir = 3  # SE
            elif next_dx == 0 and next_dy > 0:
                next_food_dir = 4  # S
            elif next_dx < 0 and next_dy > 0:
                next_food_dir = 5  # SW
            elif next_dx < 0 and next_dy == 0:
                next_food_dir = 6  # W
            elif next_dx < 0 and next_dy < 0:
                next_food_dir = 7  # NW
            else:
                next_food_dir = 0  # Default
                
            next_state_idx = (next_head_x_discrete * y_states + next_head_y_discrete) * food_directions + next_food_dir
            next_state_idx = min(next_state_idx, number_states - 1)
            

            # With this:
            if training:
                # Pass None as next_state if game is over
                next_state_for_update = None if game_over else next_state_idx
                ql.update_q_table(state_idx, action, reward, next_state_for_update)
            
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
            
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    
                pygame.display.flip()
                fps_controller.tick(difficulty)
        
        ql.save_q_table()
        print(f"Episode {episode+1}, Total reward: {total_reward}, Epsilon: {ql.epsilon:.4f}")

    
if __name__ == "__main__":
    main()

"""


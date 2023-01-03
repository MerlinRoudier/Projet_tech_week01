import gym
import numpy as np
from gym import spaces
import pygame
import time
import random


class LabyrinthEnv(gym.Env):
    def __init__(self):
        # The size of the labyrinth is 10x10
        self.size = 10

        # The agent starts at the top-left corner of the labyrinth
        self.agent_pos = [0, self.size-1]
        # The agent starts at a random position in the labyrinth
        #random.seed()
        #self.agent_pos = [random.randint(0,9), random.randint(0,9)]

        # The goal is at the bottom-right corner of the labyrinth
        self.goal_pos = [self.size-1, 0]

        # The action space is a discrete space with 4 actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # The observation space is a multi-discrete space representing the position of the agent
        self.observation_space = spaces.MultiDiscrete([self.size, self.size])

    def reset(self):
        # Reset the agent's position to the top-left corner
        self.agent_pos = [0, self.size-1]
        # Reset the agent's position to a random position
        #random.seed()
        #self.agent_pos = [random.randint(0,9), random.randint(0,9)]
        return self.agent_pos

    def step(self, action):
        # Execute the action and update the agent's position
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        elif action == 1:
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0]+1)
        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
        elif action == 3:
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1]+1)

        # Check if the agent has reached the goal
        done = np.array_equal(self.agent_pos, self.goal_pos)
        if done:
            reward = 1.0
        else:
            reward = 0.0

        return self.agent_pos, reward, done, {}

    def render(self, mode='human'):
        # Initialize pygame and the display
        pygame.init()
        
        player = pygame.image.load("robot.jpg")
        
        size_format = 80
        x=size_format*10
        y=size_format*10
        screen = pygame.display.set_mode((x, y))

        # Fill the screen with white
        screen.fill((255, 255, 255))

        # Draw the grid
        for i in range(self.size+1):
            pygame.draw.line(screen, (128, 128, 128), (i*size_format, 0), (i*size_format, y))
            pygame.draw.line(screen, (128, 128, 128), (0, i*size_format), (x, i*size_format))

        # Draw the agent and the goal
        agent_x, agent_y = self.agent_pos
        agent_x, agent_y = agent_x*size_format, agent_y*size_format
        
        goal_x, goal_y = self.goal_pos
        goal_x, goal_y = goal_x*size_format+size_format/2, goal_y*size_format+size_format/2
        
        if((agent_x == 0 and agent_y == 0)):
            screen.blit(player, (agent_x,agent_y))
            pygame.draw.rect(screen,(255,0,0),(720,0,80,80))
        elif((agent_x == 9 and agent_y == 9)):
            screen.blit(player, (agent_x,agent_y))
            pygame.draw.rect(screen,(0,255,0),(0,720,80,80))
        else:
            screen.blit(player, (agent_x,agent_y))
            pygame.draw.rect(screen,(0,255,0),(0,720,80,80))
            pygame.draw.rect(screen,(255,0,0),(720,0,80,80))
        # Update the display
        pygame.display.update()

        # Run the pygame loop until the window is closed
        #running = True
        #while running:
            #for event in pygame.event.get():
                #if event.type == pygame.QUIT:
                    #running = False


# Create the environment
env = LabyrinthEnv()

# Reset the environment
observation = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    env.render()
    #time.sleep(0.5)
    print(i,"-",action,"-",observation,"-",reward,"-",done, "-",info)
    if(done):
        break

env.close()

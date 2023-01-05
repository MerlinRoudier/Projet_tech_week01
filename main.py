import gymnasium as gym
import numpy as np
from gym import spaces
import pygame
import time
import torch
import torch.nn as nn
import torch.optim as optim


class LabyrinthEnv(gym.Env):
    def __init__(self):

        # Initialize pygame and the display
        pygame.init()        
        self.player = pygame.image.load("robot.jpg")
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
        self.directionStr = ("left", "right", "up", "down")

        # The observation space is a multi-discrete space representing the position of the agent
        self.observation_space = spaces.MultiDiscrete([self.size, self.size])
        
        self.countReward = 0

    def reset(self):
        # Reset the agent's position to the top-left corner
        self.agent_pos = [0, self.size-1]
        # Reset the agent's position to a random position
        #random.seed()
        #self.agent_pos = [random.randint(0,9), random.randint(0,9)]
        return self.agent_pos

    def step(self, action):
        # Execute the action and update the agent's position
        former_pos = np.copy(self.agent_pos)
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        elif action == 1:
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0]+1)
        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
        elif action == 3:
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1]+1)

        # punish the agent if it runs to a wall (-10pts, wall+moving cost)
        if np.array_equal(self.agent_pos, former_pos): 
            reward = -9.0
            self.countReward+=reward
        # Check if the agent has reached the goal
        done = np.array_equal(self.agent_pos, self.goal_pos)
        if done:
            reward = 1000.0
            self.countReward+=reward
        else:
            reward = -1.0
            self.countReward+=reward

        return self.agent_pos, reward, self.countReward, done, {}

    def render(self, mode='human'):
        
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
        
        #Values of the rendered rectangles must be changed if the starting and finishing points are changed
        pygame.draw.rect(screen,(0,255,0),(0,720,80,80))
        pygame.draw.rect(screen,(255,0,0),(720,0,80,80))
        screen.blit(self.player, (agent_x,agent_y))
        # Update the display
        pygame.display.update()

    def random_moves(self, iterations, rendering=True, msSteps=0):
        # Reset the environment
        observation = self.reset()
        for i in range(iterations):
            action = self.action_space.sample()
            observation, reward, countReward, done, info = self.step(action)
            if rendering: self.render()
            time.sleep(msSteps)
            print(i,"|",env.directionStr[action],"|",observation,"|",reward,"|", countReward,"|",done,"|",info)
            if(done):
                break

    def basic_moves(self, rendering=True, msSteps=0):
        observation = self.reset()
        done = False
        i=0
        while not done:            
            if rendering: self.render()
            if self.agent_pos[1]>0:
                observation, reward, countReward, done, info = self.step(2)
                action=0
            elif self.agent_pos[1] == 0:
                observation, reward, countReward, done, info = self.step(1)
                action=3
            else: done!=done
            time.sleep(msSteps)
            print(i,"|",env.directionStr[action],"|",observation,"|",reward,"|", countReward,"|",done, "|",info)





# Create the environment
env = LabyrinthEnv()

env.random_moves(1000, rendering=True, msSteps=0.1)

env.close()

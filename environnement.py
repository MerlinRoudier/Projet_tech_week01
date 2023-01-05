import gymnasium
from gymnasium import spaces 
import pygame
import time
import torch
import torch.nn as nn
import torch.optim as optim
from agent import basicAgent, randomAgent, RLAgent


class Environ(gymnasium.Env):
    def __init__(self):
        # Initialize pygame and the display
        pygame.init()        
        # The size of the labyrinth is 10x10
        self.size = 10
        # Matrix representing the environment and its rewards
        self.matrix = torch.tensor([[-1,-1,-1,-1,-1,-1,-1,-1,-1,1000],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
        #Initialize the agent arrays
        
        #The goal is at the top-right corner of the labyrinth
        self.goal_pos = torch.tensor([self.size-1, 0])

        #The starting position is at the bottom-left corner of the labyrinth
        self.starting_pos = [0, self.size-1]

        #The agent starts at the bottom-left corner of the labyrinth
        self.arrayAgent = [RLAgent(self.starting_pos[0], self.starting_pos[1], self.matrix)]

        # The action space is a discrete space with 4 actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        self.directionStr = ("left", "right", "up", "down")

        # The observation space is a multi-discrete space representing the position of the agent
        self.observation_space = spaces.MultiDiscrete([self.size, self.size])
        
        self.countReward = 0

    def reset(self):
        # Reset the agents's position to the bottom-left corner
        for a in self.arrayAgent:
            a.pos = self.starting_pos

    def step(self):
        # Execute the action and update the agent's position
        for a in self.arrayAgent:
            a.move(self.matrix)
        #return self.agent_pos, reward, self.countReward, done, {}

    def reward(self):
        # Check if the agents have reached the goals
        for a in self.arrayAgent:
            if torch.equal(a.pos, self.goal_pos):
                a.score+=1000
            else:
                a.score-=1

    def render(self, numberAgent):
        
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
        agent_x, agent_y = self.arrayAgent[numberAgent].pos
        agent_x, agent_y = agent_x*size_format, agent_y*size_format
        
        goal_x, goal_y = self.goal_pos
        goal_x, goal_y = goal_x*size_format+size_format/2, goal_y*size_format+size_format/2
        
        #Values of the rendered rectangles must be changed if the starting and finishing points are changed
        pygame.draw.rect(screen,(0,255,0),(0,720,80,80))
        pygame.draw.rect(screen,(255,0,0),(720,0,80,80))
        screen.blit(self.arrayAgent[numberAgent].image, (agent_x,agent_y))
        # Update the display
        pygame.display.update()
    
    def checkEnd(self):
        return torch.equal(self.arrayAgent[0].pos, self.goal_pos) 

    def __str__(self):
        return "pos agent: "+str(self.arrayAgent[0].pos) +" \nReward agent: "+str(self.arrayAgent[0].score)

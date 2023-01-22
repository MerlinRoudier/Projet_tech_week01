import gymnasium as gym
from gymnasium import spaces 
import pygame
import torch
from agent import basicAgent, randomAgent, RLAgent


class Environ(gym.Env):
    def __init__(self, matrix_size, goal_pos, starting_pos):
        # Initialize pygame and the display
        pygame.init()        
        # Matrix representing the environment and its rewards
        self.size = matrix_size
        self.matrix = torch.zeros(matrix_size, matrix_size)-1
        self.matrix[0,matrix_size-1]=1000 #La fin est en abscisse ?
        #Initialize the agent arrays
        
        #The goal is at the top-right corner of the labyrinth
        self.goal_pos = goal_pos

        #The starting position is at the bottom-left corner of the labyrinth
        self.starting_pos = starting_pos

        #The agent starts at the bottom-left corner of the labyrinth
        self.arrayAgent = [RLAgent(self.starting_pos[0], self.starting_pos[1], self.matrix, 0.001)]

        # The action space is a discrete space with 4 actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        self.directionStr = ("left", "right", "up", "down")

        # The observation space is a multi-discrete space representing the position of the agent
        self.observation_space = spaces.MultiDiscrete([self.size, self.size])
        
       

    def reset(self):
        # Reset the agents's position to the bottom-left corner
        for a in self.arrayAgent:
            a.pos = self.starting_pos

    def step(self):
        # Execute the action and update the agent's position
        for a in self.arrayAgent:
            a.move(self.matrix)
        #return self.agent_pos, done, {}

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
        
        # Dessiner les obstacles dans le labyrinthe.
        posx,posy = self.matrix.size()
        for i in range(posx):
            for j in range(posy):
                if(self.matrix[i][j] == -1000):
                    pygame.draw.rect(screen,(0,0,0),(i*80,j*80,80,80))
        
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

    def set_obstacle(self, coords):
        self.matrix[coords[0], coords[1]] = -1000

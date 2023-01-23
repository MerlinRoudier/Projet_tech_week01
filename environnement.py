import gymnasium as gym
from gymnasium import spaces 
import pygame
import torch
from agent import basicAgent, randomAgent, RLAgent


class Environ(gym.Env):
    #Constructor
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
        self.arrayAgent = []

        # The action space is a discrete space with 4 actions (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        self.directionStr = ("left", "right", "up", "down")

        # The observation space is a multi-discrete space representing the position of the agent
        self.observation_space = spaces.MultiDiscrete([self.size, self.size]) 

    #Agent handling part

    def add_agent(self,agent,alpha=0.85, gamma=0.3, epsilon=0.005):
        #the optionnal arguments are only required for the RL agent, choosing them while choosing another agent won't affect the process
        if agent=="RLAgent":
            self.arrayAgent.append(RLAgent(self.starting_pos[0], self.starting_pos[1],self.matrix, alpha, gamma, epsilon))
        elif agent=="Basic":
            self.arrayAgent.append(basicAgent(self.starting_pos[0], self.starting_pos[1]))
        elif agent=="Random":
            self.arrayAgent.append(randomAgent(self.starting_pos[0], self.starting_pos[1]))
        else: 
            print("required agent not found")
            exit()
        return None

    def remove_agent(self, index):
        self.arrayAgent.pop(index)
        return None 
    
    def remove_all_agents(self):
        self.arrayAgent = []
        return None
    
    def train_all_agents(self, iteration):
        #This function will train all and only the RL agents  
        #iteration is a list, that mean we want to set differents iteration by agent
        assert type(iteration) == list
        index=0
        for agent in self.arrayAgent:
            if type(agent) == RLAgent:
                agent.train(iteration[index])
                index+=1




    #Environment handling part

    def reset(self):
        # Reset the agents's position to the bottom-left corner
        for a in self.arrayAgent:
            a.pos = self.starting_pos

    def step(self):
        # Execute the action and update the agent's position
        for a in self.arrayAgent:
            a.move(self.matrix)
        #return self.agent_pos, done, {}

    def render(self):
        # Réglage de la taille de la fenêtre.
        posx,posy = self.matrix.size()
        size_format = 80
        x=size_format*posx
        y=size_format*posy
        
        screen = pygame.display.set_mode((x, y))

        # Fill the screen with white
        screen.fill((255, 255, 255))

        # Draw the grid
        for i in range(self.size+1):
            pygame.draw.line(screen, (128, 128, 128), (i*size_format, 0), (i*size_format, y))
            pygame.draw.line(screen, (128, 128, 128), (0, i*size_format), (x, i*size_format))
        
        # Dessiner les obstacles dans le labyrinthe.
        for i in range(posx):
            for j in range(posy):
                if(self.matrix[j][i] == -100000):
                    pygame.draw.rect(screen,(0,0,0),(i*size_format,j*size_format,size_format,size_format))
        
        #Values of the rendered rectangles must be changed if the starting and finishing points are changed
        pygame.draw.rect(screen,(0,255,0),(0,(size_format*10)-size_format,size_format,size_format))
        pygame.draw.rect(screen,(255,0,0),((size_format*10)-size_format,0,size_format,size_format))

        # Draw the agents 
        for i in range(len(self.arrayAgent)):
            agent_x, agent_y = self.arrayAgent[i].pos
            agent_x, agent_y = agent_x*size_format, agent_y*size_format
            screen.blit(self.arrayAgent[i].image, (agent_x,agent_y))
        # Update the display
        pygame.display.update()
    
    def checkEnd(self):
        for i in range(len(self.arrayAgent)):
            if torch.equal(self.arrayAgent[i].pos, self.goal_pos):
                return True
        return False

    def set_obstacle(self, coords):
        if(self.matrix[coords[0], coords[1]] != -100000):
            self.matrix[coords[0], coords[1]] = -100000
    
    # Misc part
    def __str__(self):
        return "pos agent: "+str(self.arrayAgent[0].pos) +" \nReward agent: "+str(self.arrayAgent[0].score)

    def close(self):
        super().close()
        pygame.quit()

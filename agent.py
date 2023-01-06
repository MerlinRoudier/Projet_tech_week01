
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


class Agent:
    def __init__(self, x, y):
        self.pos = torch.tensor([x,y])
        self.image = pygame.image.load("robot.jpg")
        self.score = 0
        self.strAction = ["up", "down", "left", "right"]
        
    def is_valid(self,matrice,pos):
        return 0 <= pos[0] < matrice.size()[0] and 0 <= pos[1] < matrice.size()[1]    


class basicAgent(Agent):
    def __init__(self, x, y):
        super().__init__(x, y)
    
    def move(self, matrice):
        if self.pos[1] == 0:
            self.pos[0]+=1
        else:
            self.pos[1]-=1
    
class randomAgent(Agent):
    def __init__(self, x, y):
        super().__init__(x, y)
    
    def move(self, matrice):
        x,y=self.pos
        r={e for e in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)] if self.is_valid(matrice,e)}.pop()
        self.pos = torch.tensor(r)


class RLAgent(Agent):
    def __init__(self, x, y, matrix):
        super().__init__(x, y)
        self.q_table = torch.zeros(4, matrix.size()[0], matrix.size()[0]) #our q-table, matrix-size multiplied by our possible actions along a new axis
        self.gamma = 0.75 #discount factor
        self.alpha = 0.9 #learning rate
        self.matrix = matrix
        self.reward = []
        
        #Train the RLAgent
        self.train(750)
        print(self.q_table)

    def q_learning(self, state, action, reward, next_state):
        #applying the Bellman formula 
        self.q_table[action, state[1], state[0]] = (1-self.alpha)*self.q_table[action, state[1], state[0]] + self.alpha*(reward+self.gamma*torch.max(self.q_table[:,next_state[1], next_state[0]]))

    def train(self, iteration):
        #saving the initial position:
        init_pos = self.pos

        #training loop to improve the q-table
        for _ in range(iteration):
            #choose an action based on the current state/current position
            action = torch.argmax(self.q_table[:,self.pos[1], self.pos[0]])
            #print("action is", action)
            action_str = self.strAction[action]

            #perform the action and recieve an award
            #hit a wall and get punished:
            hit_a_wall = False
            if action_str == "up":
                next_position = torch.tensor([self.pos[0], self.pos[1]-1])
                if not (self.is_valid(self.matrix, next_position)):
                    next_position = torch.tensor([self.pos[0], 0])
                    hit_a_wall = True

            elif action_str == "down":
                next_position = torch.tensor([self.pos[0], self.pos[1]+1])
                if not (self.is_valid(self.matrix, next_position)):
                    next_position = torch.tensor([self.pos[0], self.pos[1]])
                    hit_a_wall = True

            elif action_str == "left":
                next_position = torch.tensor([self.pos[0]-1, self.pos[1]])
                if not (self.is_valid(self.matrix, next_position)):
                    next_position = torch.tensor([0, self.pos[1]])
                    hit_a_wall = True

            elif action_str == "right":
                next_position = torch.tensor([self.pos[0]+1, self.pos[1]])
                if not (self.is_valid(self.matrix, next_position)):
                    next_position = torch.tensor([self.pos[0], self.pos[1]])
                    hit_a_wall = True
            
            self.score = self.matrix[next_position[1], next_position[0]]
            if hit_a_wall:
                self.score-=100
            self.q_learning(self.pos, action, self.score, next_position)
            self.pos = next_position
            #print("next pos is", next_position)

        #restauring everything train related for the test
        self.score=0
        self.pos = init_pos

    def move(self, _):
        #Return the index of the max element in the Q-table, corresponding to the best action to take
        print(self.pos)
        decision = torch.argmax(self.q_table[:,self.pos[1], self.pos[0]])
        if torch.equal(decision, torch.tensor(0)):
            self.pos[1]-=1
        elif torch.equal(decision, torch.tensor(1)):
            self.pos[1]+=1
        elif torch.equal(decision, torch.tensor(2)):
            self.pos[0]-=1
        elif torch.equal(decision, torch.tensor(3)):             
            self.pos[0]+=1 
        else: 
            print("wtf")  




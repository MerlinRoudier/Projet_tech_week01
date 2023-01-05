
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


class Agent:
    def __init__(self, x, y, behaviour):
        self.pos = torch.tensor([x,y])
        self.image = pygame.image.load("robot.jpg")
        self.score = 0
        self.comportement = behaviour
        #self.neural_network =

    def basic_moves(self):
        if self.pos[1] == 0:
            self.pos[0]+=1
        else:
            self.pos[1]-=1
    def is_valid(self,matrice):
        return min()

    def random_moves(self):
        x,y=self.pos
        r={e for e in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)] if self.is_valid(e)}.pop()

        i = torch.randint(1,5,(1,)) #si marche pas, faites torch.randin(1,5,(1,))
        
        if(i==1):
            self.pos[0]+=1
        elif(i==2):
            self.pos[0]-=1
        elif(i==3):
            self.pos[1]+=1
        else:
            self.pos[1]-=1
        # r=torch.randint(0,2,(2,))
        # self.pos[r[0]]+=[-1,1][r[1]]
        
    def rl_moves(self,matrice):
        return None


    def move(self, matrice):
        if(self.comportement == "basic"):
            self.basic_moves()
        elif(self.comportement == "random"):
            self.random_moves()    
        elif(self.comportement == "rl"):
            self.rl_moves(matrice)


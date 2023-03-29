import torch
from uuid import uuid4
from os import path, makedirs

class LRLAgent:
	def __init__(self, pos, size, alpha, gamma, epsilon):
		self.pos=torch.tensor(pos)
		self.origin=self.pos
		self.size=size
		self.gamma=gamma
		self.alpha=alpha
		self.epsilon=epsilon
		self.weights = torch.rand((4,1))/100
		self.features = torch.tensor((size*2,),dtype=torch.float)
		self.q_values = torch.matmul(self.weights, self.features)

	def move(self,sim=False):
		if not sim and torch.rand(1)<self.epsilon:
			return int(torch.randint(low=0,high=4, size=(1,)))
		return int(torch.argmax(self.q_values))

	def update(self, action, reward, new_state):
		self.features=torch.tensor((-reward,),dtype=torch.float)
		next_q_values=torch.max(torch.matmul(self.weights, self.features)) 
		print("diff: ",self.gamma*next_q_values-self.q_values)
		for i in range(len(self.weights[action])):
			self.weights[action][i] = self.weights[action][i]+self.alpha*(reward+self.gamma*next_q_values-self.q_values)*float((self.features[i])/self.features.norm())
		self.q_values=next_q_values
		print(self.weights)


def setup(typeAgent, pos, size, alpha, gamma, epsilon):
	if typeAgent=='basic':
		return basicAgent(pos)
	elif typeAgent=='random':
		return randomAgent(pos)
	elif typeAgent=='rl':
		return RLAgent(pos, size, alpha,gamma,epsilon)
	elif typeAgent=='lrl':
		return LRLAgent(pos, size, alpha, gamma, epsilon)
	else:
		raise Exception("Invalid agent choice")

import torch
from uuid import uuid4
from os import path, makedirs


class basicAgent:
	def __init__(self, pos):
		self.pos=torch.tensor(pos)
		self.origin=self.pos

	def move(self,sim=None):
		return 0 if self.pos[0]<9 else 1


class randomAgent:
	def __init__(self, pos):
		self.pos=torch.tensor(pos)
		self.origin=self.pos
		
	def move(self,sim=None):
		return int(torch.randint(low=0, high=4, size=(1,)))


class RLAgent:
	def __init__(self, pos, size, alpha, gamma, epsilon):
		self.pos=torch.tensor(pos)
		self.origin=self.pos
		self.q_table = torch.rand(4, size, size)/100
		self.gamma=gamma
		self.alpha=alpha
		self.epsilon=epsilon

	def move(self,sim=False):
		if not sim and torch.rand(1)<self.epsilon:
			return int(torch.randint(low=0,high=4, size=(1,)))
		return int(torch.argmax(self.q_table[:,self.pos[0], self.pos[1]]))
        	
	def update(self, action, reward, new_pos):
		self.q_table[action, self.pos[0], self.pos[1]]= \
		(1-self.alpha)*self.q_table[action,self.pos[0],self.pos[1]] + \
		self.alpha*(reward+self.gamma*torch.max(self.q_table[:,new_pos[0],new_pos[1]]))

	def save_q_table(self):
		if not path.isdir('q_table_saves'): makedirs('q_table_saves')
		torch.save(self.q_table, path.join('q_table_saves', str(uuid4())+"_q_table.pt"))
	
	def load_q_table(self):
		self.q_table = torch.load("q_table.pt")

class LRLAgent:
	def __init__(self, pos, size, alpha, gamma, epsilon):
		self._pos=torch.tensor(pos)
		self.origin=self.pos
		self.size=size
		self.states = torch.zeros(size*size).unsqueeze(0)
		#self.weights = torch.rand((size*size,4))/100
		#self.bias = torch.rand(4)/100
		self.weights = torch.rand((4,3))/100
		self.gamma=gamma
		self.alpha=alpha
		self.epsilon=epsilon
		self.dico={pos:0}

		x1 = self.size-pos[0]+self.size-pos[1]
		self.features = torch.tensor((x1,0,0), dtype=torch.float)

	def update_s(self,new_pos):
		self.states[0,self.pos[0]+self.size*self.pos[1]]=0
		self.states[0,new_pos[0]+self.size*new_pos[1]]=1

	@property
	def pos(self):
		return self._pos

	@pos.setter
	def pos(self,new_pos):
		self.update_s(new_pos)  
		self._pos=new_pos
	
	def move(self,sim=False):
		result = torch.matmul(self.weights, self.features)

		if not sim and torch.rand(1)<self.epsilon:
			return int(torch.randint(low=0,high=4, size=(1,)))
		action=int(torch.argmax(result))
		return action
	
	def update_features(self,new_state):
		self.features[0]=(self.size-new_state[0]+self.size-new_state[1]) #x1
		if (n:=tuple(new_state.tolist())) in self.dico: 
			self.dico[n]+=1
			self.features[1] = self.dico[n] #x2
		else: 
			self.dico.update({n:0})
			self.features[1]=0
		self.features[2]+=1

	def update(self, action, reward, new_state):
		formerMax = torch.max(torch.matmul(self.weights, self.features))

		# self.features[0]=(self.size-new_state[0]+self.size-new_state[1]) #x1
		# if (n:=tuple(new_state.tolist())) in self.dico: 
		# 	self.dico[n]+=1
		# 	self.features[1] = self.dico[n] #x2
		# else: 
		# 	self.dico.update({n:0})
		# 	self.features[1]=0
		# self.features[2]+=1 #x3
		#self.features/=self.features.norm()*100 #features normalizing
		self.update_features(new_state)

		#sauvegarder les poids ailleurs pour la MAJ
		currentMax = torch.max(torch.matmul(self.weights, self.features)) #/self.features.norm()
		print("diff: ",self.gamma*currentMax-formerMax)
		for i in range(len(self.weights[action])):
			self.weights[action][i] = self.weights[action][i]+self.alpha*(reward+self.gamma*currentMax-formerMax)*float((self.features[i])/self.features.norm())
			#print(f"feature {i}: {self.features[i]}")
			#print("norm: ",float(self.features[i]/self.features.norm()))
		#print("inside alpha", (reward+self.gamma*currentMax-formerMax))
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

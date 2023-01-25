import torch
import gymnasium as gym
import pygame
from time import sleep
from agent import setup

class Env(gym.Env):
	def __init__(self, size=10, timeout=1000, nb_i=1000, rendering='visual', goal_pos=(9,9), obstacles=[]):
		pygame.init()
		self.size=size
		self.timeout=timeout
		self.nb_i=nb_i
		self.obstacles=obstacles
		self.rendering=rendering
		self.goal_pos=torch.tensor(goal_pos)
		self.time=0
		self.agents=[]
		self.image=pygame.image.load("robot.jpg")
        
		self._action_to_direction = {
			0: torch.tensor((1,0)),
			1: torch.tensor((0,1)),
			2: torch.tensor((-1,0)),
			3: torch.tensor((0,-1)),
		}

	def add_agent(self,typeAgent='basic', pos=(0,0), alpha=.85, gamma=.3, epsilon=.01):
		self.agents.append(setup(typeAgent,pos,self.size,alpha,gamma,epsilon))

	def has_ended(self, is_alive=True):
		if self.time>self.timeout or not is_alive:
			return True
		for agent in self.agents:
			if torch.equal(agent.pos, self.goal_pos):
				return True
		return False

	def render_visual(self,pos=torch.tensor((-1,-1))):
		posx,posy = self.size,self.size
		size_format = 80
		x=size_format*posx
		y=size_format*posy

		screen = pygame.display.set_mode((x, y))

		screen.fill((255, 255, 255))

		for i in range(self.size+1):
			pygame.draw.line(screen, (128, 128, 128), (i*size_format, 0), (i*size_format, y))
			pygame.draw.line(screen, (128, 128, 128), (0, i*size_format), (x, i*size_format))

		for i in range(posx):
			for j in range(posy):
				if((i,j) in self.obstacles):
					pygame.draw.rect(screen,(0,0,0),(j*size_format,(self.size-i-1)*size_format,size_format,size_format))

		pygame.draw.rect(screen,(0,255,0),(0,(size_format*self.size)-size_format,size_format,size_format))
		pygame.draw.rect(screen,(255,0,0),((size_format*self.size)-size_format,0,size_format,size_format))

		for agent in self.agents:
			agent_x, agent_y = agent.pos
			agent_x, agent_y = (self.size-agent_x-1)*size_format, agent_y*size_format
			screen.blit(self.image, (agent_y,agent_x))
		pygame.display.update()
		sleep(.5)

	def render_tty(self):
		s=''
		for i in range(self.size-1, -1, -1):
			for j in range(0, self.size):
				square=torch.tensor((i,j))
				if tuple(square) in [tuple(agent.pos) for agent in self.agents]:
					s+='R '
				elif torch.equal(square, self.goal_pos):
					s+='E '
				elif tuple(square) in self.obstacles:
					s+='O '
				else:
					s+='* '
			s+='\n'
		for agent in self.agents:
			s+=str(agent.pos)
			s+='\n'
		sleep(.5)
		print(s)

	def render(self):
		if self.rendering=='visual':
			self.render_visual()
		elif self.rendering=='tty':
			self.render_tty()

	def start(self):
		self.reset()
		for agent in self.agents:
			agent.pos=torch.tensor((0,0))
		while not self.has_ended():
			for agent in self.agents:
				action=agent.move()
				new_pos	,reward, is_alive=self.step(agent.pos, action)
				if self._is_valid(new_pos):
					agent.pos=new_pos
				self.time+=1
			self.render()

	def train(self, num_agent=0):
		agent=self.agents[num_agent]
		for _ in range(self.nb_i):
			self.reset()
			agent.pos=torch.tensor((0,0))
			is_alive=True
			while not self.has_ended(is_alive):
				action=agent.move()
				new_pos,reward, is_alive = self.step(agent.pos, action)
				if(is_alive):
					agent.update(action, reward, new_pos)
					agent.pos=new_pos
					self.time+=1
				else:
					agent.update(action, reward, agent.pos)

	def reset(self):
		self.time=0
		
	def _is_valid(self, pos):
		return ((pos[0]>=0 and pos[0]<self.size) and (pos[1]>=0 and pos[1]<self.size)) and (tuple(pos) not in self.obstacles)
	
	def step(self, pos, action):
		is_alive=True
		new_pos=pos+self._action_to_direction[action]
		if torch.equal(new_pos,self.goal_pos):
			reward=1e3
			is_alive=False
		elif not self._is_valid(new_pos):
			reward=-1e3
			is_alive=False
		else:
			reward=-1
			self.time+=1
			if self.time>=self.timeout:
				is_alive=False
		return new_pos, reward, is_alive


import torch
import gymnasium as gym
import pygame
from time import sleep
from agent import setup
from PIL import Image

class Env(gym.Env):
	def __init__(self, size=10, timeout=1000, rendering='visual', goal_pos=(9,9), obstacles=[]):
		pygame.init()
		self.size=size
		self.timeout=timeout
		self.obstacles=obstacles
		self.rendering=rendering
		self.goal_pos=torch.tensor(goal_pos)
		self.time=0
		self.agents=[]
        
		self._action_to_direction = {
			0: torch.tensor((1,0)),
			1: torch.tensor((0,1)),
			2: torch.tensor((-1,0)),
			3: torch.tensor((0,-1)),
		}

		# REQUIRED FOR THE RENDER VISUAL, ONLY REQUIRED TO INITIALIZE ONCE
		self.x,self.y = 500,500 # static pixels size for the window
		self.size_format = int(self.x/self.size) #dynamique
		image = Image.open('robot.jpg')
		n_image = image.resize((self.size_format,self.size_format))
		n_image.save('robot.jpg')

		self.image=pygame.image.load("robot.jpg")


	def add_agent(self,typeAgent='basic', pos=(0,0), alpha=.85, gamma=.3, epsilon=.01):
		self.agents.append(setup(typeAgent,pos,self.size,alpha,gamma,epsilon))

	def has_ended(self, is_alive=True):
		if self.time>self.timeout or not is_alive:
			return True
		for agent in self.agents:
			if torch.equal(agent.pos, self.goal_pos):
				return True
		return False

	def render_visual(self,refresh):

		screen = pygame.display.set_mode((self.x, self.y))

		screen.fill((255, 255, 255))

		for i in range(self.size+1):
			pygame.draw.line(screen, (128, 128, 128), (i*self.size_format, 0), (i*self.size_format, self.y))
			pygame.draw.line(screen, (128, 128, 128), (0, i*self.size_format), (self.x, i*self.size_format))

		for i in range(self.size):
			for j in range(self.size):
				if((i,j) in self.obstacles):
					pygame.draw.rect(screen,(0,0,0),(j*self.size_format,(self.size-i-1)*self.size_format,self.size_format,self.size_format))

		for agent in self.agents:
			pygame.draw.rect(screen,(0,255,0),(agent.origin[0]*self.size_format,(self.size-agent.origin[1]-1)*self.size_format,self.size_format,self.size_format))		
		pygame.draw.rect(screen,(255,0,0),(self.goal_pos[0]*self.size_format,(self.size-self.goal_pos[1]-1)*self.size_format,self.size_format,self.size_format))

		for agent in self.agents:
			agent_x, agent_y = agent.pos
			agent_x, agent_y = (self.size-agent_x-1)*self.size_format, agent_y*self.size_format
			screen.blit(self.image, (agent_y,agent_x))
		pygame.display.update()
		sleep(refresh)

	def render_tty(self, refresh):
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
		sleep(refresh)
		print(s)

	def render(self, refresh=.5):
		if self.rendering=='visual':
			self.render_visual(refresh)
		elif self.rendering=='tty':
			self.render_tty(refresh)

	def start(self):
		self.reset()
		for agent in self.agents:
			agent.pos=agent.origin
		stopped = False
		while not self.has_ended() and not stopped:
			for agent in self.agents:
				action=agent.move()
				new_pos	, _, _ =self.step(agent.pos, action)
				if self._is_valid(new_pos):
					agent.pos=new_pos
			self.time+=1
			self.render()
			stopped = self.quit()


	def train(self, num_agent=0, nb_i=1000):
		agent=self.agents[num_agent]
		for _ in range(nb_i):
			self.reset()
			agent.pos=agent.origin
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
			reward=0
		return new_pos, reward, is_alive
	
	def quit(self):
		if self.rendering == 'visual':
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.display.quit()                
					pygame.quit()
					return True
			return False
		
		


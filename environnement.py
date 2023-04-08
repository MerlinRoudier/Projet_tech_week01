import torch
import gymnasium as gym
import pygame
from time import sleep
from agent import setup
from PIL import Image
from os import path
from agent import LRLAgent


class Env(gym.Env):
    def __init__(self,
                 size: int = 10,
                 timeout: int = 1000,
                 rendering: str = 'visual',
                 goal_pos: tuple = (9, 9),
                 obstacles: list = []) -> None:
        self.size = size
        self.timeout = timeout
        self.obstacles = obstacles
        self.rendering = rendering
        self.goal_pos = torch.tensor(goal_pos)
        self.time = 0
        self.agents = []
        self._action_to_direction = {
            0: torch.tensor((1, 0)),
            1: torch.tensor((0, 1)),
            2: torch.tensor((-1, 0)),
            3: torch.tensor((0, -1)),
        }
        if rendering == 'visual':
            self.setup_visual()

    def add_agent(self,
                  typeAgent='basic',
                  pos=(0, 0),
                  alpha=.85,
                  gamma=.3,
                  epsilon=.01):
        self.agents.append(setup(typeAgent,
                                 pos,
                                 self.size,
                                 alpha,
                                 gamma,
                                 epsilon))

    def has_ended(self, is_alive: bool = True) -> bool:
        if self.time > self.timeout or not is_alive:
            return True
        for agent in self.agents:
            if torch.equal(agent.pos, self.goal_pos):
                return True
        return False

    def setup_visual(self) -> None:
        # REQUIRED FOR THE RENDER VISUAL
        pygame.init()
        self.x, self.y = 500, 500  # static pixels size for the window
        self.size_format = int(self.x / self.size)  # dynamic
        self.image = []
        # checking if images are generated in the
        # cache for the designed matrix size and screen size
        is_gen = False
        for i in range(10):
            try:
                image = Image.open(path.join('__pycache__',
                                             str(i)+'_robot.png'))
            except Exception:
                is_gen = False
                break
            if image.size[0] == image.size[1] and \
               image.size != self.size_format:
                is_gen = False
                image.close()
        # if the images aren't in the cache or well sized, we resize and regen
        if not is_gen:
            for i in range(10):
                image = Image.open(path.join('robots_img',
                                             str(i)+'_robot.png'))
                n_image = image.resize((self.size_format,
                                        self.size_format))
                n_image.save(path.join('__pycache__',
                                       str(i)+'_robot.png'))
                image.close()
                n_image.close()
                self.image.append(pygame.image.load(
                    path.join('__pycache__',
                              str(i)+'_robot.png')))

    def render_visual(self, refresh: float) -> None:
        screen = pygame.display.set_mode((self.x, self.y))
        screen.fill((255, 255, 255))
        for i in range(self.size+1):
            pygame.draw.line(screen, (128, 128, 128),
                             (i*self.size_format, 0),
                             (i*self.size_format, self.y))
            pygame.draw.line(screen, (128, 128, 128),
                             (0, i*self.size_format),
                             (self.x, i*self.size_format))
        for i in range(self.size):
            for j in range(self.size):
                if ((i, j) in self.obstacles):
                    pygame.draw.rect(screen, (0, 0, 0),
                                     (j*self.size_format,
                                      (self.size-i-1)*self.size_format,
                                      self.size_format, self.size_format))
        for agent in self.agents:
            pygame.draw.rect(screen, (0, 255, 0),
                             (agent.origin[0]*self.size_format,
                              (self.size-agent.origin[1]-1)*self.size_format,
                              self.size_format, self.size_format))
        pygame.draw.rect(screen, (255, 0, 0),
                         (self.goal_pos[0]*self.size_format,
                          (self.size-self.goal_pos[1]-1)*self.size_format,
                          self.size_format, self.size_format))
        for i in range(len(self.agents)):
            agent_x, agent_y = self.agents[i].pos
            agent_x = (self.size-agent_x-1)*self.size_format
            agent_y = agent_y*self.size_format
            screen.blit(self.image[i % 10], (agent_y, agent_x))
        pygame.display.update()
        sleep(refresh)

    def render_tty(self, refresh: float) -> None:
        s = ''
        for i in range(self.size-1, -1, -1):
            for j in range(0, self.size):
                square = torch.tensor((i, j))
                if tuple(square) in [tuple(agent.pos)
                                     for agent in self.agents]:
                    s += 'R '
                elif torch.equal(square, self.goal_pos):
                    s += 'E '
                elif tuple(square) in self.obstacles:
                    s += 'O '
                else:
                    s += '* '
            s += '\n'
        for agent in self.agents:
            s += str(agent.pos)
            s += '\n'
        sleep(refresh)
        print(s)

    def render(self, refresh: float = .5) -> None:
        if self.rendering == 'visual':
            self.render_visual(refresh)
        elif self.rendering == 'tty':
            self.render_tty(refresh)

    def start(self) -> None:
        self.reset()
        for agent in self.agents:
            agent.pos = agent.origin
        stopped = False
        while not self.has_ended() and not stopped:
            for agent in self.agents:
                action = agent.move(True)
                new_pos, _, _ = self.step(agent, agent.pos, action)
                if self._is_valid(new_pos):
                    agent.pos = new_pos
            self.time += 1
            self.render()
            stopped = self.quit()

    def train(self,
              num_agent: int = 0,
              nb_i: int = 1000,
              immortal: bool = False) -> None:
        agent = self.agents[num_agent]
        for _ in range(nb_i):
            self.reset()
            agent.pos = agent.origin
            is_alive = True
            while not self.has_ended(is_alive):
                action = agent.move()
                new_pos, reward, is_alive = self.step(agent, agent.pos, action)
                if (is_alive):
                    agent.update(action, reward, new_pos)
                    agent.pos = new_pos
                else:
                    agent.update(action, reward, agent.pos)
                if immortal:
                    is_alive = True

    def reset(self) -> None:
        self.time = 0

    def _is_valid(self, pos: torch.Tensor) -> bool:
        return ((pos[0] >= 0 and pos[0] < self.size) and
                (pos[1] >= 0 and pos[1] < self.size)) and \
                 (tuple(pos) not in self.obstacles)

    def step(self, agent, pos: torch.Tensor, action: int) -> tuple:
        is_alive = True
        new_pos = pos + self._action_to_direction[action]
        reward = agent.reward(new_pos,
                              self.goal_pos,
                              (valid_move := self._is_valid(new_pos)))
        is_alive = valid_move
        return new_pos, reward, is_alive

    def gen_maze(self, pos: tuple = (0, 0)) -> None:
        maze = torch.ones(self.size+1, self.size+1)
        search = [(pos[0]+1, pos[1]+1)]
        path = set()
        goal_pos = search[0]
        while search:
            pos = search[-1]
            maze[pos] = 0
            verif = True
            tries = 4
            while tries and verif:
                rand = int(torch.randint(0, tries, size=(1,)))
                new_pos = (pos[0]+self._action_to_direction[rand][0],
                           pos[1]+self._action_to_direction[rand][1])
                x, y = pos
                if rand == 0:
                    sight = maze[x+1:x+3, y-1:y+2]
                elif rand == 1:
                    sight = maze[x-1:x+2, y+1:y+3]
                elif rand == 2:
                    sight = maze[x-2:x, y-1:y+2]
                else:
                    sight = maze[x-1:x+2, y-2:y]
                if self._is_valid(new_pos) and \
                   sight.sum() >= 6 and \
                   new_pos not in path:
                    search.append(new_pos)
                    path.add(new_pos)
                    verif = False
                tries -= 1
            if verif:
                goal_pos = max(goal_pos, search.pop(-1))
        self.obstacles = [(i-1, j-1)
                          for i in range(1, self.size+1)
                          for j in range(1, self.size+1) if maze[i, j] == 1]
        self.goal_pos = torch.tensor((goal_pos[0]-1, goal_pos[1]-1))

    def quit(self) -> bool:
        if self.rendering == 'visual':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    return True
            return False

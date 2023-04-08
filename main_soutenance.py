from environnement import Env

basicEnv=Env()
basicEnv.add_agent(typeAgent='basic')
basicEnv.start()

input()

randomEnv=Env(size=11,timeout=20, goal_pos=(10,10))
randomEnv.add_agent(typeAgent='random', pos=(5,5))
randomEnv.start()

input()

RLEnv=Env()
RLEnv.add_agent(typeAgent='rl')
RLEnv.train()
RLEnv.start()

input()

RLEnvMaze=Env()
RLEnvMaze.gen_maze()
RLEnvMaze.add_agent(typeAgent='rl')
RLEnvMaze.train()
RLEnvMaze.start()

input()

RLEnvBigMaze=Env()
RLEnvBigMaze.gen_maze()
RLEnvBigMaze.add_agent(typeAgent='rl')
RLEnvBigMaze.agents[0].load_q_table()
RLEnvBigMaze.start()

input()

LRLEnv=Env()
LRLEnv.add_agent(typeAgent='lrl', alpha=.3, gamma=.1, epsilon=.3)
LRLEnv.train()
LRLEnv.start()
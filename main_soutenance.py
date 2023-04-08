from environnement import Env
from lab import obstacles,goal_pos

def simulation(nb_env):
    match nb_env:
        case 0:
            basicEnv=Env()
            basicEnv.add_agent(typeAgent='basic')
            basicEnv.start()

        case 1:
            randomEnv=Env(size=11,timeout=20, goal_pos=(10,10))
            randomEnv.add_agent(typeAgent='random', pos=(5,5))
            randomEnv.start()

        case 2:
            RLEnv=Env()
            RLEnv.add_agent('rl')
            RLEnv.train()
            RLEnv.start()

        case 3:
            MultiEnv=Env()
            MultiEnv.add_agent('rl')
            MultiEnv.add_agent('basic')
            MultiEnv.add_agent('random')
            MultiEnv.train()
            MultiEnv.start()

        case 4:
            RLEnvMaze=Env()
            RLEnvMaze.gen_maze()
            RLEnvMaze.add_agent(typeAgent='rl')
            RLEnvMaze.train()
            RLEnvMaze.start()

        case 5:
            RLEnvBigMaze=Env(size=30,goal_pos=goal_pos)
            RLEnvBigMaze.obstacles=obstacles
            RLEnvBigMaze.add_agent(typeAgent='rl')
            RLEnvBigMaze.agents[0].load_q_table()
            RLEnvBigMaze.start()

        case 6:
            LRLEnv=Env(size=5, goal_pos=(4,4), timeout=20)
            LRLEnv.add_agent(typeAgent='lrl', alpha=.3, gamma=.1, epsilon=.3)
            LRLEnv.train(nb_i=1000)
            LRLEnv.start()

i=int(input())
while i<7:
   simulation(i)
   i=int(input())

# env=Env(size=50)
# env.gen_maze()
# with open("lab3.py","w") as f:
#     f.write("obstacles="+str(env.obstacles)+"\n")
#     f.write("goal_pos="+str((int(env.goal_pos[0]),int(env.goal_pos[1])))+"\n")
# env.add_agent(typeAgent='rl')
# env.train(nb_i=1000)
# env.agents[0].save_q_table()
# env.start()
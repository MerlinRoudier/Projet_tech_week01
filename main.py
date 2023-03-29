from environnement import Env
import agent
import torch

def set_obstacles(s,size):
    o=[]
    t=s.split(' ')
    for i in range(size):
        for j in range(size):
            if t[size*i+j]=='O':
                o+=[(i,j)]
    return o


def decisionMapping(agent: agent.LRLAgent, size: int, goalpos : tuple, startpos: tuple) -> None:
    d = {
        0: "up",
        1: "right",
        2: "down",
        3: "left"
    }
    total = list()
    for i in range(size):
        tmp = list()
        for j in range(size):
            features = torch.tensor([i/goal_pos[0], j/goal_pos[1], i+j/sum(goal_pos)])
            action = d[int(torch.argmax(torch.matmul(agent.weights, features)))]
            tmp.append(action)
        total.append(tmp)
    for i in range(len(total)):
        for j in range(len(total[0])):
            print(total[i][j], end=" ")
        print("")





o='\
* * * * * O * * * * \
O O O O * O * O * * \
* O * O * O O O O * \
* * * * * * * O * * \
O O O O * O O O * * \
* * * * * O * * * * \
* O O * O O * O * * \
* O * * * O * O O * \
* O O O O O * O * * \
* * * * * * * O O *'
#o=set_obstacles(o,10)
goal_pos=(4,4)
env=Env(size=5, rendering='visual', goal_pos=goal_pos)
env.add_agent(typeAgent='lrl', alpha=.35, gamma=.1, epsilon=.3)
#env.gen_maze()
decisionMapping(env.agents[0], 5, (4,4), (0,0))
env.train(nb_i=500)
#env.agents[0].save_q_table()
env.start()
decisionMapping(env.agents[0], 5, (4,4), (0,0))




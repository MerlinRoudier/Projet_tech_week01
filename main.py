import torch
from environnement import Env

def set_obstacles(s,size):
    o=[]
    t=s.split(' ')
    for i in range(size):
        for j in range(size):
            if t[size*i+j]=='O':
                o+=[(i,j)]
    return o

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
o=set_obstacles(o,10)

env=Env(obstacles=o, rendering='tty')
env.add_agent('rl')
env.train()
env.agents[0].save_q_table()
env.start()


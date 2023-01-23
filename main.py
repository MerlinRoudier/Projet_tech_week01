import time
from environnement import Environ 
from agent import *
import torch 

# Create the environment matrix
matrix_size = 10 #squarred matrix
env = Environ(matrix_size, torch.tensor([matrix_size-1, 0]), torch.tensor([0, matrix_size-1]))
env.add_agent("RLAgent", alpha=0.275, gamma=0.3, epsilon=0.3)
iteration=1000

for i in range(30):
    env.set_obstacle(torch.randint(0,9,(2,1)))
env.train_all_agents([iteration])

#env.arrayAgent[0].save_q_table()
#env.arrayAgent[0].load_q_table()

for i in range(1000):
    env.step()
    env.render()
    time.sleep(0.5)
    print(f"testing iteration number {i+1}")
    if(env.checkEnd()): break

env.close()

import time
from environnement import Environ 
import torch 

# Create the environment matrix
matrix_size = 10 #squarred matrix
env = Environ(matrix_size, torch.tensor([matrix_size-1, 0]), torch.tensor([0, matrix_size-1]))
#iteration =1500
# env.arrayAgent[0].train(iteration)
# env.arrayAgent[0].save_q_table()
env.arrayAgent[0].load_q_table()

for _ in range(1000):
    env.step()
    env.render(0)
    time.sleep(0.5)
    if(env.checkEnd()): break

env.close()

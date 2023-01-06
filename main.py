import time
from environnement import Environ  

# Create the environment
env = Environ()

for _ in range(1000):
    env.step()
    env.reward()
    env.render(0)
    time.sleep(0.5)
    #print(env)
    if(env.checkEnd()): break

env.close()

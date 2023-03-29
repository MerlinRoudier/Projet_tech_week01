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
#o=set_obstacles(o,10)
goal_pos=(4,4)
env=Env(size=5, rendering='visual', goal_pos=goal_pos)
env.add_agent(typeAgent='lrl', alpha=.25, gamma=.1, epsilon=.3)
#env.gen_maze()
env.train(nb_i=500)
#env.agents[0].save_q_table()
env.start()


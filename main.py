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
O O O O * * * O * * \
* O * O * O O O O * \
* * * * * * * O * * \
O O O O * O O O * * \
* * * * * O * * * * \
* O O * O O * O * * \
* O * * * * * O O * \
* O O O O O * O * * \
* * * * * * * O O *'
o=set_obstacles(o,10)
goal_pos=(9,9)
env=Env(size=10, rendering='visual', goal_pos=goal_pos, obstacles=o)
env.add_agent(typeAgent='random')
env.add_agent(typeAgent='random')
env.add_agent(typeAgent='random')
env.add_agent(typeAgent='random')
#env.add_agent(typeAgent='rl')
#env.gen_maze()
#env.train(num_agent=1, nb_i=1000)
#env.agents[0].save_q_table()
env.start()


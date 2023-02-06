import torch
import numpy as np
import random
from environnement import Env

'''def set_obstacles(s,size):
    o=[]
    t=s.split(' ')
    for i in range(size):
        for j in range(size):
            if t[size*i+j]=='O':
                o+=[(i,j)]
    return o

"""o='\
* * * * * O * * * * \
O * * * * O * * * * \
* O * * * O * * * * \
* * * * * * * * * * \
O O O O * * * * * * \
* * * * * * * * * * \
* * * * * * * O * * \
* * * * * * * O O * \
* * * * * * * O * * \
* * * * * * * O O *'
o=set_obstacles(o,10)"""
'''

def Maze():
    def preprocess_grid(grid:np.ndarray, size:int) -> np.ndarray:
    # fix first row and last column to avoid digging outside the maze external borders
        first_row = grid[0]
        first_row[first_row == 1] = 0
        grid[0] = first_row
        for i in range(1,size):
            grid[i,size-1] = 1
        return grid

    def carve_maze(grid:np.ndarray, size:int) -> np.ndarray:
        output_grid = np.empty([size*3, size*3],dtype=str)
        output_grid[:] = '#'

        i = 0
        j = 0
        while i < size:
            w = i*3 + 1
            while j < size:
                k = j*3 + 1
                toss = grid[i,j]
                output_grid[w,k] = ' '
                if toss == 0 and k+2 < size*3:
                    output_grid[w,k+1] = ' '
                    output_grid[w,k+2] = ' '
                if toss == 1 and w-2 >=0:
                    output_grid[w-1,k] = ' '
                    output_grid[w-2,k] = ' '

                j = j + 1

            i = i + 1
            j = 0

        return output_grid

    def conv(grid,size):
        o=[]
        for i in range((size*3)):
            for j in range((size*3)):
                if grid[i][j]=='#':
                    o+=[(i,j)]
        return o

    def estDansTableau1(element,liste):
        for i in range(len(liste)):
            if liste[i] == element:
                liste.pop(i)
                return liste
        return liste

    n=1
    p=0.3 #Plus p augmente, plus cela prends de temps à l'entrainer et il a moins de chance de trouver un chemin viable
    size=4

    #np.random.seed(42)
    grid = np.random.binomial(n,p, size=(size,size))
    processed_grid = preprocess_grid(grid, size)
    output = carve_maze(processed_grid, size)
    tab=conv(output,size)

    tabVal=[(0,0),((3*size)-1,(3*size)-1)]
    a=random.randint(1,2)
    if(a==1):
        tabVal.append((0,1))
    else:
        tabVal.append((1,0))
    a=random.randint(1,2)
    if(a==1):
        tabVal.append(((3*size)-2,(3*size)-1))
    else:
        tabVal.append(((3*size)-1,(3*size)-2))
    for i in range(len(tabVal)):
        tab=estDansTableau1(tabVal[i],tab)

    return tab

o=Maze()


env=Env(obstacles=o,size=12,rendering='visual', goal_pos=(11,11), timeout=5000)
env.add_agent('rl')
env.train()
env.start()


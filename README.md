# Projet_tech_week01

The goal of this project is the development of a multi-agent environment using the Pytorch module.
This repository is developped in collaboration with Meta as part of a computer engineering technological project of EILCO.

Firstly, we define an environment as a X*X matrix where each cell can be set as a wall or a path. The walls disposition and the matrix size can be set as the user wishes. Each cell has a different reward. So far an empty cell representing a path that an agent can cross has a reward of 0, a wall has a reward of -1000 and the try to get out of the environment by the agent is also rewarded by -1000. The goal position also known as the maze exit has a reward of 1000. The environment supports multiple agents nesting different algorithms listed below. 

We will consider a context where an agent will move in an environment and where each of those movements has an impact on the environment. The actions of the agent trigger a reward at the end, and he has to maximise this reward.

To begin our journey, we defined four types of agent:

- basic
- random
- RL
- LRL

The basic agent follows a single type of behaviour: he goes up until he can't anymore, then he move to the right. He isn't aware of rewards and need to be in a fenceless maze to progress.

The random agent follow a purely random behavior, selecting a random decision wherever he is in the environment. He isn't aware of rewards too and may finally end up to the goal position depending on the environment (size, complexity etc.).

The RL (reinforcement learning) agent embed a Q-Table making him the first interesting agent to work with. He update its Q-Table using the Bellman Formula, is aware of rewards and need training iterations to properly update its Q-table (according to its learning rate, its discount rate, the reward etc.). Once trained in a specific environment, he is able to reach the goal position thanks to its Q-table making him converge toward the solution. Thus there exist some limitations to this approach. The RL agent can't find the goal position in a complex and big environment.  

The LRL (Linear Reinforcement Learning) agent gets rid of the Q-table by adding a layer of abstraction. This abstraction will further be used to implement a neural network and generalizes the concept of evolving through different environments. This agent will also be aware of the rewards and requires a training.

Training will consist of iterations. During an iteration we let an agent evolve in an environment and use rewards to guide himself toward the goal position. The agent will have a defined number of maximum steps to do. If this maximum is reached, or the goal position is reached or even if the agent hit a wall (including getting out of the environment), the Q-table is updated and the iteration end.

---

## Requirements 

This program requires multiple modules:
- Gymnasium
- Pytorch
- PIL
- Pygame

To run the program, simply edit and run main.py 
```Python
python3 main.py
``` 
on Linux or 
```Python 
py main.py
```
on Windows

--- 

## To-Do List

We are still currently working on this project, here are the unordered differents things to implement or patch:

- [ ] Add a vebose mode 
- [ ] Implement the Linear Reinforcement Learning Agent
- [ ] Create a different reward system (Manhattan or Euclidian distance)
- [ ] Compare 2 Reinforcement Learning agent training where one cross walls, the other not 
- [ ] Having a variable epsilon (randomness term) through training
- [ ] Having a variable alpha (learning rate) through training
- [ ] getting rid of the epsilon (randomness term) during the testing stage 
- [ ] Making a notebook handling the code to show some metrics within matplotlib

---

## Disclaimer

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

--- 

All credits to Benjamin Bicrel (Enyobii), Merlin Roudier (Edenlawen), Matthias Wyss (Mattha212) and Romain Levifve (RMI78) 
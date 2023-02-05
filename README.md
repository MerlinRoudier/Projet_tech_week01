# Projet_tech_week01

The goal of this project is the development of a series of reinforcement learning algorithms. We will consider a context where an agent will move in an environment and where each of those movements has an impact on the environment. The actions of the agent trigger a reward at the end, and he has to maximise this reward.
This repository is developped in collaboration with Meta as part of a computer engineering technological project of EILCO.

We defined four types of agent:

- basic
- random
- RL
- LRL

The basic agent follows a single type of behaviour: he goes up until he can't anymore, then he move to the right. For each movement the agent will be rewarded by a negative value (-1), and by a +1000 when he arrives at the end.
The random agent follows a purely random behaviour.

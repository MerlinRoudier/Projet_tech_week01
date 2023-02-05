# Projet_tech_week01

The goal of this project is the development of a series of reinforcement learning algorithms. We will consider a context where an agent will move in an environment and where each of those movements has an impact on the environment. The actions of the agent trigger a reward at the end, and he has to maximise this reward.
This repository is developped in collaboration with Meta as part of a computer engineering technological project of EILCO.

Le projet a pour but de développer une série d'algorithmes d'apprentissages par renforcement.
On considèrera un contexte où un agent se déplacera dans un environnement où chacune de ses actions fait varier
le dit environnement. Ces actions causent également la reception d'une récompense, et l'agent doit maximiser cette récompense.
Ce repository est développé en collaboration avec Meta dans le cadre d'un Projet Tech d'ingénieur informatique à l'Eilco.

Nous avons défini quatre types d'agents:
We defined four types of agent:

- basic
- random
- RL
- LRL

The basic agent follow a single type of behaviour: he goes up until he can't anymore, then he move to the right. For each movement the agent will be rewarded by a negative value (-1), and by a +1000 when he arrives at the end.
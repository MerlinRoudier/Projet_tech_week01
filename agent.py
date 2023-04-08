import torch
from uuid import uuid4
from os import path, makedirs


class basicAgent:
    def __init__(self, pos: tuple) -> None:
        self.pos = torch.tensor(pos)
        self.origin = self.pos

    def move(self, sim: bool = False) -> int:
        return 0 if self.pos[0] < 9 else 1

    def reward(self, new_pos, goal_pos, isValid):
        pass


class randomAgent:
    def __init__(self, pos: tuple) -> None:
        self.pos = torch.tensor(pos)
        self.origin = self.pos

    def move(self, sim: bool = False) -> int:
        return int(torch.randint(low=0, high=4, size=(1,)))

    def reward(self, new_pos, goal_pos, isValid):
        pass


class RLAgent:
    def __init__(self,
                 pos: tuple,
                 size: int,
                 alpha: float,
                 gamma: float,
                 epsilon: float) -> None:
        self.pos = torch.tensor(pos)
        self.origin = self.pos
        self.q_table = torch.rand(4, size, size)/100
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def move(self, sim: bool = False) -> int:
        if not sim and torch.rand(1) < self.epsilon:
            return int(torch.randint(low=0, high=4, size=(1,)))
        return int(torch.argmax(self.q_table[:, self.pos[0], self.pos[1]]))

    def update(self, action: int, reward: float, new_pos: torch.Tensor):
        self.q_table[action, self.pos[0], self.pos[1]] = \
            (1-self.alpha)*self.q_table[action, self.pos[0], self.pos[1]] + \
            self.alpha*(reward+self.gamma*torch.max(self.q_table[:,
                                                                 new_pos[0],
                                                                 new_pos[1]]))

    def reward(self,
               new_pos: torch.Tensor,
               goal_pos: torch.Tensor,
               isValid: bool) -> float:
        if torch.equal(new_pos, goal_pos):
            return 1e3
        elif not isValid:
            return -1
        else:
            return 0

    def save_q_table(self) -> None:
        if not path.isdir('q_table_saves'):
            makedirs('q_table_saves')
        torch.save(self.q_table, path.join('q_table_saves',
                                           str(uuid4())+"_q_table.pt"))

    def load_q_table(self) -> None:
        self.q_table = torch.load("q_table.pt")


class LRLAgent:
    def __init__(self,
                 pos: tuple,
                 size: int,
                 alpha: float,
                 gamma: float,
                 epsilon: float) -> None:
        self.pos = torch.tensor(pos)
        self.origin = self.pos
        self.size = size
        self.states = torch.zeros(size*size).unsqueeze(0)
        self.weights = torch.rand((4, 3))/10
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        x1 = self.size - pos[0]
        x2 = self.size - pos[1]
        self.features = torch.tensor((x1, x2, x1+x2), dtype=torch.float)

    def move(self, sim: bool = False) -> int:
        if sim:
            self.features = self.update_features(self.pos)
        result = torch.matmul(self.weights, self.features)
        if not sim and torch.rand(1) < self.epsilon:
            return int(torch.randint(low=0, high=4, size=(1,)))
        action = int(torch.argmax(result))
        return action

    def update_features(self, new_state: torch.Tensor) -> torch.Tensor:
        # copy the attributes
        features = self.features.clone().detach()

        # compute the features
        features[0] = (new_state[0]) / self.size
        features[1] = (new_state[1]) / self.size
        features[2] = (features[1]+features[2]) / self.size * 2
        return features

    def update(self,
               action: int,
               reward: float,
               new_state: torch.Tensor) -> None:
        formerMax = torch.max(torch.matmul(self.weights, self.features))
        self.features = self.update_features(new_state)
        currentMax = torch.max(torch.matmul(self.weights, self.features))
        for i in range(len(self.weights[action])):
            self.weights[action][i] = \
                self.weights[action][i] + \
                self.alpha * (reward+self.gamma*currentMax-formerMax) * \
                float((self.features[i]))

    def reward(self,
               new_pos: torch.Tensor,
               goal_pos: torch.Tensor,
               isValid: bool) -> float:
        features = self.update_features(new_pos)
        if torch.equal(new_pos, goal_pos):
            return float(torch.sum(features*torch.tensor((1, 1, 1))))*10
        else:
            return float(torch.sum(features*torch.tensor((1, 1, 1))))


def setup(typeAgent: str,
          pos: tuple,
          size: int,
          alpha: float,
          gamma: float,
          epsilon: float):
    if typeAgent == 'basic':
        return basicAgent(pos)
    elif typeAgent == 'random':
        return randomAgent(pos)
    elif typeAgent == 'rl':
        return RLAgent(pos, size, alpha, gamma, epsilon)
    elif typeAgent == 'lrl':
        return LRLAgent(pos, size, alpha, gamma, epsilon)
    else:
        raise Exception("Invalid agent choice")

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.lin(x))


class BackModel(nn.Module):
    def __init__(self, num_classes=24*6, num_resnets=3, num_skips=2):
        super().__init__()
        self.input = nn.Linear(33, 64)
        self.relu = nn.ReLU()
        self.num_skips = num_skips

        self.resnets = nn.ModuleList()

        for _ in range(num_skips):
            group = nn.Sequential(*[ResidualBlock() for _ in range(num_resnets)])
            self.resnets.append(group)

        self.move = nn.Linear(64, num_classes)
        self.evaluator = nn.Linear(64, 1)

        for layer in [self.evaluator, self.move]:
            nn.init.xavier_uniform_(layer.weight)


    def forward(self, x):
        x = self.relu(self.input(x))  

        for group in self.resnets:
            x = x + group(x)
                          
        move = torch.log_softmax(self.move(x), dim=1)
        eval = torch.tanh(self.evaluator(x))
                  
        return move, eval
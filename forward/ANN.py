import torch.nn as nn
import torch


# Create ANN Model
class ANNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output1_dim, output2_dim):
        super(ANNModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.Tanh()  # ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.Tanh()  # ReLU()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.Tanh()  # ReLU()

        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.Tanh()  # ReLU()

        self.fc5_1 = nn.Linear(hidden_dim, output1_dim)
        self.tanh1 = nn.Tanh()  # nn.Sigmoid()

        self.fc5_2 = nn.Linear(hidden_dim, output2_dim)
        self.tanh2 = nn.Tanh()  # nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.relu4(out)

        out1 = self.fc5_1(out)
        out1 = self.tanh1(out1)

        out2 = self.fc5_2(out)
        out2 = self.tanh2(out2)

        out = torch.cat((out1, out2), dim=1)

        return out

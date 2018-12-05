# Model
import torch.nn as nn
from torch.nn import functional as F
class MLP_Regression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP_Regression, self).__init__()

        linear1_output_len = 120
        linear2_output_len = 84
        linear3_output_len = num_classes

        self.linear1 = nn.Linear(in_features=input_size, out_features=linear1_output_len)
        self.linear2 = nn.Linear(in_features=linear1_output_len, out_features=linear2_output_len)
        self.linear3 = nn.Linear(in_features=linear2_output_len, out_features=linear3_output_len)

    def forward(self, x):
        y1 = F.relu(self.linear1(x))
        y2 = F.relu(self.linear2(y1))
        out = self.linear3(y2)

        return out
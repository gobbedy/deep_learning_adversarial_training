# Model
import torch.nn as nn
from torch.nn import functional as F
class MLP_Regression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP_Regression, self).__init__()

        factor=10
        linear1_output_len = 120*factor
        linear2_output_len = 84*factor
        linear3_output_len = 120*factor
        linear4_output_len = 84*factor
        linear5_output_len = 120*factor
        linear6_output_len = num_classes

        self.linear1 = nn.Linear(in_features=input_size, out_features=linear1_output_len)
        self.linear2 = nn.Linear(in_features=linear1_output_len, out_features=linear2_output_len)
        self.dropout1 = nn.Dropout(p=0.7)
        self.linear3 = nn.Linear(in_features=linear2_output_len, out_features=linear3_output_len)
        self.dropout2 = nn.Dropout(p=0.7)
        self.linear4 = nn.Linear(in_features=linear3_output_len, out_features=linear4_output_len)
        self.linear5 = nn.Linear(in_features=linear4_output_len, out_features=linear5_output_len)
        self.dropout3 = nn.Dropout(p=0.7)
        self.linear6 = nn.Linear(in_features=linear5_output_len, out_features=linear6_output_len)

        #self.linear = nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x):
        # y1 = F.relu(self.linear1(x))
        # #y1_drop = self.dropout1(y1)
        # #y2 = F.relu(self.linear2(y1_drop))
        # y2 = F.relu(self.linear2(y1))
        # y2_drop = self.dropout2(y2)
        # out = self.linear3(y2_drop)

        y1 = F.relu(self.linear1(x))
        y1_drop = self.dropout1(y1)
        y2 = F.relu(self.linear2(y1_drop))
        y3 = F.relu(self.linear3(y2))
        y2_drop = self.dropout2(y3)
        y4 = F.relu(self.linear4(y2_drop))
        y5 = F.relu(self.linear5(y4))
        y3_drop = self.dropout3(y5)
        out = self.linear6(y3_drop)

        # y1 = F.relu(self.linear1(x))
        # y2 = F.relu(self.linear2(y1))
        # y3 = F.relu(self.linear3(y2))
        # y4 = F.relu(self.linear4(y3))
        # y5 = F.relu(self.linear5(y4))
        # out = self.linear6(y5)

        #out = self.linear(x)

        return out
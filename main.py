from typing import Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            device: Optional[str],
            fc_hidden_size: int = 128,
    ):
        super().__init__()
        self.device = device or 'cpu'
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size, device=device)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=fc_hidden_size, device=device)
        self.fc2 = nn.Linear(in_features=fc_hidden_size, out_features=output_size, device=device)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h_0 = Variable(torch.zeros(1, seq_len, self.hidden_size).to(self.device))
        c_0 = Variable(torch.zeros(1, seq_len, self.hidden_size).to(self.device))

        output, (final_h, final_c) = self.lstm(x, (h_0, c_0))
        # output = (batch_size, seq_len, hidden_dim)
        # final_h = (1, seq_len, fc_hidden_size)
        output_fc1 = self.fc1(final_h)
        # output_fc1 = (1, seq_len, fc_hidden_size)
        output_fc2 = self.fc2(output_fc1)
        # output_fc2 = (1, seq_len, 6)
        return output_fc2



SEQUENCE_LEN = 1000
BATCH_SIZE = 16
HIDDEN_DIM = 8
HIDDEN_FC_DIM = 128
OUTPUT_CLASSES = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_tensor = torch.from_numpy(np.random.randn(BATCH_SIZE, SEQUENCE_LEN, 6))
input_tensor = input_tensor.float().to(device)

model = LSTMClassifier(hidden_size=32,
                       output_size=OUTPUT_CLASSES,
                       device=device)

output = model(input_tensor)

print(output)
print(output.shape)


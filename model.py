from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            device: Optional[str],
            fc_hidden_size: int = 128,
            n_layers: int = 2,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.device = device or 'cpu'
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=6,
                            hidden_size=hidden_size,
                            dropout=dropout,
                            num_layers=n_layers,
                            device=device)
        self.fc1 = nn.Linear(in_features=hidden_size * n_layers,
                             out_features=fc_hidden_size,
                             device=device)
        self.fc2 = nn.Linear(in_features=fc_hidden_size,
                             out_features=output_size,
                             device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h_0 = Variable(torch.zeros(self.n_layers, seq_len, self.hidden_size).to(self.device))
        c_0 = Variable(torch.zeros(self.n_layers, seq_len, self.hidden_size).to(self.device))

        output, (final_h, final_c) = self.lstm(x, (h_0, c_0))
        # output = (batch_size, seq_len, hidden_dim)
        # final_h = (n_layers, seq_len, fc_hidden_size)

        unpacked = final_h.permute(1, 0, 2).reshape(seq_len, -1)
        # unpacked = (seq_len, n_layers * fc_hidden_size)
        hidden = self.dropout(unpacked)

        output_fc1 = self.fc1(hidden)
        # output_fc1 = (seq_len, fc_hidden_size)
        output_fc2 = self.fc2(output_fc1)
        output_fc2 = self.dropout(output_fc2)
        # output_fc2 = (seq_len, 6)
        return output_fc2

import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__() 
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # output = [ batch_size, seq_len, hidden_dim * 2] 
        output, (hidden, cell) = self.rnn(x)

        hidden = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)  # [1, batch_size, hidden_dim * 2]
        cell = torch.cat((cell[0], cell[1]), dim=1).unsqueeze(0)        # [1, batch_size, hidden_dim * 2]

        return output, (hidden, cell)

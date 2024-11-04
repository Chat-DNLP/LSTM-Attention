import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__() 
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # [ batch_size, seq_len, hidden_dim * 2] 
        output, (hidden, cell) = self.rnn(x)

        print(output.shape)
        print(hidden.shape)
        # [ num_layers, batch size, hidden_dim]
        hidden = self._combine_directions(hidden)

        print(hidden.shape)

        cell = self._combine_directions(cell)
        
        return output, (hidden, cell)

    def _combine_directions(self, states):

        forward_state = states[0::2]
        backward_state = states[1::2]
        return torch.cat((forward_state, backward_state), dim=2)
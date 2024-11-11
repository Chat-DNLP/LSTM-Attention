import torch.nn as nn
import torch
from attention.attention_factory import AttentionFactory

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.attention = AttentionFactory.initialize_attention("Bilinear", hidden_dim, hidden_dim)


    def forward(self, x, hidden, cell, outputs_encoder):
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        
        attention_vectors = self.attention(hidden, outputs_encoder)

        hidden_attention = hidden.transpose(0, 1)

        output_attention = torch.cat((attention_vectors, hidden_attention), dim=2)
        output_attention = self.linear(output_attention)
        output_luong = torch.tanh(output_attention)

        output = self.fc_out(output_luong)
        return output, (hidden, cell)
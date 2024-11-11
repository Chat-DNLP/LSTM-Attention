import torch.nn as nn
import torch 
from attention.attention_factory import AttentionFactory

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim*2, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2*2, hidden_dim*2) # 512 * 2(bidireccional) * 2 (concatenación)
        self.fc_out = nn.Linear(hidden_dim*2, output_dim)
        self.attention = AttentionFactory.initialize_attention("Multi-Layer Perceptron", hidden_dim*2, hidden_dim*2)

    def forward(self, x, hidden, cell, outputs_encoder):

        output, (hidden, cell) = self.rnn(x, (hidden, cell))

        attention_vectors = self.attention(hidden, outputs_encoder)

        hidden_attention = hidden.transpose(0, 1)
        output_attention = torch.cat((attention_vectors, hidden_attention), dim=2)

        output_attention = self.linear(output_attention)
        output = self.fc_out(output_attention)

        return output, (hidden, cell)
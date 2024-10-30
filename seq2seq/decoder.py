import torch.nn as nn
import torch 

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, attention):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) 
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.attention = attention

    # Modificar el forward para que haga la atención
    def forward(self, x, hidden, cell, outputs_encoder):
        output, (hidden, cell) = self.rnn(x, (hidden, cell))

        # attention_weights -> [ batch_size = 8, 1 valor, X palabras en el encoder = 3] -> [ 8, 3]
        attention_weights = self.attention.compute_score(output, outputs_encoder)

        # Normalized vectors -> [ 8, 3, 1]
        normalized_vectors = torch.softmax(attention_weights, dim=1).unsqueeze(-1)

        # [ 8, 3, 512] * [ 8, 3, 512] = [8, 3, 512]
        attention_output = normalized_vectors * outputs_encoder

        # Promedio de los vectores -> [8, 1, 512]
        summed_vectors = torch.sum(attention_output, dim=1, keepdim=True)

        # output = [8,1,512]
        output = self.fc_out(summed_vectors)
        # output = [8,1, tamaño_vocab]
        
        return output, (hidden, cell)
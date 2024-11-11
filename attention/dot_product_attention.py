import torch.nn as nn
import torch

class DotProductAttention(nn.Module):

    def __init__(self, decoder_hidden_dim, encoder_dim):
        super().__init__()

    def forward(self, hidden_state, encoder_states):
        h_t = hidden_state.squeeze(0)  # [batch, embedding_dim]
        h_t = h_t.unsqueeze(2)  # [batch, embedding_dim, 1]
        score = torch.bmm(encoder_states, h_t)  # Resultado es [batch, seq_len, 1]
        attention_weights = score.squeeze(2)

        # Normalized vectors -> [ 8, 3, 1]
        normalized_vectors = torch.softmax(attention_weights, dim=1).unsqueeze(-1)

        # [ 8, 3, 512] * [ 8, 3, 512] = [8, 3, 512]
        attention_output = normalized_vectors * encoder_states

        # Promedio de los vectores -> [8, 1, 512]
        summed_vectors = torch.sum(attention_output, dim=1, keepdim=True)

        return summed_vectors
import torch.nn as nn
import torch

class DotProductAttention(nn.Module):

    def __init__(self, decoder_hidden_dim, encoder_dim):
        super().__init__()

    def forward(self, hidden_state, encoder_states):
        h_t = hidden_state.squeeze(0)  # [batch, embedding_dim]
        h_t = h_t.unsqueeze(2)  # [batch, embedding_dim, 1]
        score = torch.bmm(encoder_states, h_t)  # Resultado es [batch, seq_len, 1]

        return score.squeeze(2)
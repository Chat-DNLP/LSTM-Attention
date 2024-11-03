import torch.nn as nn
import torch

class BilinearAttention(nn.Module):

    def __init__(self, decoder_hidden_dim, encoder_dim):
        super().__init__()
        self.W = nn.Linear(decoder_hidden_dim, encoder_dim)

    def forward(self, hidden_state, encoder_states):
        hidden_proj = self.W(hidden_state)

        attention_scores = torch.bmm(encoder_states, hidden_proj.unsqueeze(2)).squeeze(2)
        return attention_scores
        
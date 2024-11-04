import torch.nn as nn
import torch

class BilinearAttention(nn.Module):

    def __init__(self, decoder_hidden_dim, encoder_dim):
        super().__init__()
        self.W = nn.Linear(decoder_hidden_dim, encoder_dim, bias=False)

    def forward(self, hidden_state, encoder_states):
        # encoder_states: [batch_size, num_words, encoder_dim]
        # hidden_state: [1, batch_size, decoder_dim]

        hidden_proj = self.W(hidden_state) # [1, batch_size, decoder_hidden_dim]
        hidden_proj = hidden_proj.squeeze(0) # [batch_size, encoder_dim]
        hidden_proj = hidden_proj.unsqueeze(2) # [batch_size, encoder_dim, 1]

        attention_scores = torch.bmm(encoder_states, hidden_proj)
        return attention_scores.squeeze(2)
        
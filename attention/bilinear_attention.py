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
        attention_weights = attention_scores.squeeze(2)

        # Normalized vectors -> [ 8, 3, 1]
        normalized_vectors = torch.softmax(attention_weights, dim=1).unsqueeze(-1)

        # [ 8, 3, 512] * [ 8, 3, 512] = [8, 3, 512]
        attention_output = normalized_vectors * encoder_states

        # Promedio de los vectores -> [8, 1, 512]
        summed_vectors = torch.sum(attention_output, dim=1, keepdim=True)

        return summed_vectors

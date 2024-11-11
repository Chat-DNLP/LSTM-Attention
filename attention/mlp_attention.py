import torch
import torch.nn as nn

class MLPAttention(nn.Module):

    def __init__(self, decoder_hidden_dim, encoder_dim):
        super().__init__()
        self.W1 = nn.Linear(decoder_hidden_dim + encoder_dim, encoder_dim, bias=False)
        self.W2 = nn.Linear(encoder_dim, 1, bias=False)
  
    def forward(self, decoder_hidden_state, encoder_states):
        # encoder_states: [batch_size, num_words, encoder_dim]
        # decoder_state: [1, batch_size, decoder_dim]

        decoder_hidden_state = decoder_hidden_state.permute(1, 0, 2) # [batch_size, 1, decoder_hidden_dim]
        decoder_state_expanded = decoder_hidden_state.expand(-1, encoder_states.size(1), -1)

        concatenated = torch.cat((decoder_state_expanded, encoder_states), dim=2)

        tanh_output = torch.tanh(self.W1(concatenated)) 

        attention_weights = self.W2(tanh_output).squeeze(2)

        normalized_vectors = torch.softmax(attention_weights, dim=1).unsqueeze(-1)
        attention_output = normalized_vectors * encoder_states
        summed_vectors = torch.sum(attention_output, dim=1, keepdim=True)

        return summed_vectors

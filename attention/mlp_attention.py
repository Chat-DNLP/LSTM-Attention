import torch
import torch.nn as nn

class MLPAttention():

    def __init__(self):
        self.W1 = None
        self.W2 = None
  
    def compute_score(self, decoder_state, encoder_state):
        # encoder_states: [batch_size, num_words, encoder_dim]
        # decoder_state: [batch_size, 1, decoder_dim]

        decoder_state_dim = decoder_state.size(-1)
        encoder_state_dim = encoder_state.size(-1)

        if self.W1 is None and self.W2 is None:
            self.W1 = nn.Linear(decoder_state_dim + encoder_state_dim, encoder_state_dim, bias=False)
            self.W2 = nn.Linear(encoder_state_dim, 1, bias=False)

        decoder_state_expanded = decoder_state.expand(-1, encoder_state.size(1), -1)
        # [ batch = 8, num = 3, dim = 512*2]
        concat_states = torch.cat((encoder_state, decoder_state_expanded), dim=-1)

        score_tanh = torch.tanh(self.W1(concat_states))
        attention_scores = self.W2(score_tanh).squeeze(-1)

        return attention_scores

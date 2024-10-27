import torch
import torch.nn as nn

class MLPAttention():

    @staticmethod
    def compute_score(decoder_state, encoder_state):

        decoder_state_dim = decoder_state.size(-1)
        encoder_state_dim = encoder_state.size(-1)

        hidden_dim = (decoder_state_dim + encoder_state_dim) // 2

        W1 = nn.Linear(decoder_state_dim + encoder_state_dim, hidden_dim, bias=False)
        W2 = nn.Linear(hidden_dim, 1, bias=False)

        extended_decoder= decoder_state.unsqueeze(2).expand(-1, -1, encoder_state.size(1), -1)
        extended_encoder = encoder_state.unsqueeze(1).expand(-1, decoder_state.size(1), -1, -1)
        concatenated = torch.cat((extended_decoder, extended_encoder), dim=-1)

        score = W2(torch.tanh(W1(concatenated))).squeeze(-1)

        return score

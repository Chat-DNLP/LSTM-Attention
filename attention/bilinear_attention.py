import torch
import torch.nn as nn

class BilinearAttention:

    @staticmethod
    def compute_score(decoder_state, encoder_state):

        decoder_state_dim = decoder_state.size(-1)
        encoder_state_dim = encoder_state.size(-1)

        bilinear = nn.Linear(decoder_state_dim, encoder_state_dim, bias=False)

        transformed_decoder_state = bilinear(decoder_state)
        return torch.matmul(transformed_decoder_state, encoder_state.transpose(-2, -1))
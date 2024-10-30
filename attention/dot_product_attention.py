import torch

class DotProductAttention:

    def compute_score(self, decoder_state, encoder_state):
        return torch.matmul(decoder_state, encoder_state.transpose(1, 2)).squeeze()
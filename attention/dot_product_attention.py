import torch

class DotProductAttention:

    @staticmethod
    def compute_score(decoder_state, encoder_state):
        return torch.matmul(decoder_state, encoder_state.transpose(-2, -1))
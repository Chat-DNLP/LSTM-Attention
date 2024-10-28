import torch

class DotProductAttention:

    @staticmethod
    def compute_score(decoder_state, encoder_state):
        encoder_state_t = encoder_state.transpose(1, 2)
        product = torch.matmul(decoder_state, encoder_state_t)
        return product.squeeze()
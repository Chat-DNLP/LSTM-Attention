import torch

class DotProductAttention:

    def compute_score(self, decoder_state, encoder_state):
        encoder_state_t = encoder_state.transpose(1, 2)
        product = torch.matmul(decoder_state, encoder_state_t)
        return product.squeeze()
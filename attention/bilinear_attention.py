import torch.nn as nn
import torch

class BilinearAttention:

    def __init__(self):
        self.W = None

    def compute_score(self, decoder_state, encoder_state):

        decoder_dim = decoder_state.size(-1)
        encoder_dim = encoder_state.size(-1)

        if self.W is None:
            self.W = nn.Parameter(torch.randn(decoder_dim, encoder_dim))

        output_W = torch.matmul(decoder_state, self.W)
        attention_scores = torch.bmm(output_W, encoder_state.transpose(1, 2))

        return attention_scores.squeeze(1)
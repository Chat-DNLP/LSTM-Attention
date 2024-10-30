import torch.nn as nn
import torch

# debe heredar de pytroch patra que se ajuste
class BilinearAttention(nn.Module):

    def __init__(self):
        super(BilinearAttention, self).__init__()
        self.W = None

    def compute_score(self, hidden_state, encoder_states):

        print("hidden_state", hidden_state.shape)
        print("encoder_states", encoder_states.shape)

        decoder_dim = hidden_state.size(-1)
        encoder_dim = encoder_states.size(-1)

        if self.W is None:
            self.W = nn.Parameter(torch.randn(decoder_dim, encoder_dim))

        output_W = torch.matmul(hidden_state, self.W)
        attention_scores = torch.bmm(output_W, encoder_states.transpose(1, 2))

        return attention_scores.squeeze(1)
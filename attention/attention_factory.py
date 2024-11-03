from attention.dot_product_attention import DotProductAttention
from attention.bilinear_attention import BilinearAttention
from attention.mlp_attention import MLPAttention

class AttentionFactory:

    __functions = {
        "Dot-product": DotProductAttention,
        "Bilinear": BilinearAttention,
        "Multi-Layer Perceptron": MLPAttention
    }

    @staticmethod
    def initialize_attention(key, decoder_hidden_dim, encoder_dim):
        attention_class = AttentionFactory.__functions.get(key)
        if attention_class is not None:
            return attention_class(decoder_hidden_dim, encoder_dim)
        return None


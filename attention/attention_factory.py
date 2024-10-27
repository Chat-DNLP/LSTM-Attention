from attention.dot_product_attention import DotProductAttention
from attention.bilinear_attention import BilinearAttention
from attention.mlp_attention import MLPAttention

class AttentionFactory:

    __functions = {
        "Dot-product": DotProductAttention(),
        "Bilinear": BilinearAttention(),
        "Multi-Layer Perceptron": MLPAttention()
    }

    @staticmethod
    def initialize_attention(key):
        functions_class = AttentionFactory.__functions.get(key)
        if functions_class is not None:
            return functions_class
        return None


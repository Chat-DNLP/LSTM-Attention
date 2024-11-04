import torch
import torch.nn as nn
import torchtext
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder                           
        self.es_embeddings = torchtext.vocab.FastText(language='es')
        self.M = self.es_embeddings.vectors
        self.M = torch.cat((self.M, torch.zeros((4, self.M.shape[1]))), 0)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        target_len = target.shape[1]
        batch_size = target.shape[0]

        # Tensor para almacenar las salidas del decoder
        outputs = torch.zeros(batch_size, target_len, 985671)
        
        # Primero, la fuente es procesada por el encoder
        outputs_encoder, (hidden, cell) = self.encoder(source)

        # La primera entrada al decoder es el vector <sos>
        x = target[:, 0, :]

        for t in range(1, target_len):
            output, (hidden, cell) = self.decoder(x.unsqueeze(1), hidden, cell, outputs_encoder)
            outputs[:, t, :] = output.squeeze(1)
            
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force:
                x = target[:, t, :]
            else:
                x = torch.matmul(output.squeeze(1), self.M)
        return outputs
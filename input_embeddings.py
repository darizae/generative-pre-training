import torch
import torch.nn as nn

class WordEmbeddings(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 weight_init_params: tuple):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self._init_weights(weight_init_params)

    def forward(self, x: torch.tensor):
        return self.embedding(x)
    
    def _init_weights(self, weight_init_params):
        # Initialize weights from a normal distribution N(0, 0.02)
        mean, std = weight_init_params
        nn.init.normal_(self.embedding.weight, mean, std)
    

class PositionalEncodings(nn.Module):

    def __init__(self,
                 context_window: int,
                 d_model: int):
        
        super().__init__()

        self.pe = nn.Embedding(
            num_embeddings=context_window,
            embedding_dim=d_model
        )

    def forward(self, x: torch.tensor):
        return self.pe(x)
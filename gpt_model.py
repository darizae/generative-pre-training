import torch
import torch.nn as nn

from config import Config
from decoder import DecoderLayer
from input_embeddings import PositionalEncodings, WordEmbeddings

class GPT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.word_embeddings = WordEmbeddings(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            weight_init_params=config.weight_init_params
        )

        self.positional_encodings = PositionalEncodings(
            context_window=config.context_window,
            d_model=config.d_model
        )

        self.decoders = []
        for _ in range(config.num_of_layers):
            decoder_block = DecoderLayer(config=config)
            self.decoders.append(decoder_block)
        self.decoders = nn.ModuleList(self.decoders)

        self.dropout = nn.Dropout(p=config.dropout_prob)

        self.config = config

    def forward(self, x: torch.tensor):
        batch_size, context_window = x.shape

        # Get word embeddings
        word_embeddings = self.word_embeddings(x)

        # Apply positional encodings
        positions = torch.arange(0, context_window).expand(batch_size, context_window).to(self.config.device) 
        word_embeddings = word_embeddings + self.positional_encodings(x=positions)

        # Apply dropout before passing through decoder block
        decoder_output = self.dropout(word_embeddings)

        # Pass through decoder layers
        for decoder_layer in self.decoders:
            decoder_output = decoder_layer(decoder_output)

        return decoder_output
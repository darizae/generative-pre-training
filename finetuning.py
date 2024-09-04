import torch
import torch.nn as nn

from config import Config
from gpt_model import GPT

# The Language Model/Text Prediction Head
class LMHead(nn.Module):

    def __init__(self, config: Config, gpt: GPT):

        super().__init__()

        self.gpt = gpt

        self.predict_next_token = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False
        )
        
        # Weight tying
        self.predict_next_token.weight = self.gpt.word_embeddings.embedding.weight

    def forward(self, x: torch.tensor):

        # Pass the input through the generative pre-trained layers
        gpt_decoder_output = self.gpt(x)

        # Predict next token in the sequence
        logits = self.predict_next_token(gpt_decoder_output)

        return logits
    

# The Classification Head
class CLSHead(nn.Module):

    def __init__(self, config: Config, gpt: GPT):
        super().__init__()

        self.gpt = gpt

        self.predict_next_token = nn.Linear(
            in_features=config.d_model,
            out_features=config.vocab_size,
            bias=False
        )

        # Weight tying
        self.predict_next_token.weight = self.gpt.word_embeddings.embedding.weight

        self.classifier = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_class
        )

        # Initialize the weight of the classifier
        _, std = config.weight_init_params
        nn.init.normal_(self.classifier.weight, std)

    def forward(self, x: torch.tensor):

        # Pass the input through the generative pre-trained layers
        gpt_decoder_output = self.gpt(x)

        # Predict next token in the sequence
        lm_logits = self.predict_next_token(gpt_decoder_output)

        # Classify based on logits
        cls_logits = self.classifier(gpt_decoder_output)

        return lm_logits, cls_logits


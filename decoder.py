import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

class MultiheadAttention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 dropput_prob: float,
                 num_of_heads: int,
                 weight_init_params: tuple,
                 ):
        super().__init__()

        self.qkv_projection = nn.Linear(
            in_features=d_model,
            out_features=d_model * 3
        )

        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=d_model
        )

        self.dropout = nn.Dropout(p=dropput_prob)

        self.num_of_heads = num_of_heads
        self.head_dim = d_model // num_of_heads

        self._init_weights(weight_init_params)

    def _init_weights(self, weight_init_params: tuple):
        # Initialize weights from a normal distribution N(0, 0.02)
        mean, std = weight_init_params
        nn.init.normal_(self.qkv_projection.weight, mean=mean, std=std)
        nn.init.normal_(self.output_projection.weight, mean=mean, std=std)

        # Initialize biases to zero
        nn.init.constant_(self.qkv_projection.bias, 0.0)
        nn.init.constant_(self.output_projection.bias, 0.0)

    def forward(self, 
            x: torch.tensor,
            mask: torch.tensor = None
            ) -> torch.tensor:
    
        batch_size, seq_len, d_model = x.shape

        # Project linearly to get Q, K and V matrices
        qkv = self.qkv_projection(x)

        # Prepare to split third dimension in three parts (Q, K and V)
        # Each of these three will operate within a defined head with a defined dimensionality
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_of_heads, self.head_dim)

        # Split tensor along Q, K and V
        # Have dimension '3' first to index by it
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Tensors are ready to be unpacked
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention
        dot_product = torch.matmul(query, key.transpose(-2, -1))
        scaling_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=query.device))
        attention_scores = torch.div(dot_product, scaling_factor)

        # Apply mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # Get probability distribution
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate the heads
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        # Apply final linear projection
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)

        return attention_output



class FeedForwardModule(nn.Module):
    def __init__(self,
                 d_model: int,
                 ffn_hidden_size: int,
                 dropout_prob: float,
                 weight_init_params: tuple):
        super().__init__()

        self.layer_one = nn.Linear(
            in_features=d_model,
            out_features=ffn_hidden_size
        )

        self.layer_two = nn.Linear(
            in_features=ffn_hidden_size,
            out_features=d_model
        )

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_prob)

        self._init_weights(weight_init_params)

    def _init_weights(self, weight_init_params):
        # Initialize weights from a normal distribution N(0, 0.02)
        mean, std = weight_init_params
        nn.init.normal_(self.layer_one.weight, mean=mean, std=std)
        nn.init.normal_(self.layer_two.weight, mean=mean, std=std)

        # Initialize biases to zero
        nn.init.constant_(self.layer_one.bias, 0.0)
        nn.init.constant_(self.layer_two.bias, 0.0)

    def forward(self, x: torch.tensor) -> torch.tensor:
        
        # (batch, seq_len, d_model) --> (batch, seq_len, ffn_hidden_size)
        x = self.layer_one(x)

        # Activation function
        x = self.gelu(x)

        # Apply dropout
        x = self.dropout(x)

        # (batch, seq_len, ffn_hidden_size) --> (batch, seq_len, d_model)
        x = self.layer_two(x)

        # Apply dropout again
        x = self.dropout(x)

        return x


class LayerNormalization(nn.Module):
    def __init__(self,
                 d_model: int):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: torch.tensor, sublayer_output: torch.tensor) -> torch.tensor:

        # Add residual connection
        residual_connection = x + sublayer_output

        # Apply layer normalization
        normalized_layer = self.layer_norm(residual_connection)

        return normalized_layer

class DecoderLayer(nn.Module):
    def __init__(self,
                 config: Config):
        super().__init__()

        self.multihead_attention = MultiheadAttention(
            d_model=config.d_model,
            dropput_prob=config.dropout_prob,
            num_of_heads=config.num_of_heads,
            weight_init_params=config.weight_init_params
        )

        self.layer_norm_one = LayerNormalization(
            d_model=config.d_model
        )

        self.feed_forward_module = FeedForwardModule(
            d_model=config.d_model,
            ffn_hidden_size=config.ffn_hidden_size,
            dropout_prob=config.dropout_prob,
            weight_init_params=config.weight_init_params
        )

        self.layer_norm_two = LayerNormalization(
            d_model=config.d_model
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        
        batch_size, seq_len, _ = x.shape
        context_window = seq_len  # Use seq_len as the context window

        mask = self._make_mask(batch_size, context_window)

        attention_output = self.multihead_attention(
            x=x,
            mask=mask
        )

        add_norm_one = self.layer_norm_one(
            x=x,
            sublayer_output=attention_output
        )

        ff_output = self.feed_forward_module(
            x=add_norm_one
        )
        
        add_norm_two = self.layer_norm_two(
            x=add_norm_one,
            sublayer_output=ff_output
        )

        return add_norm_two

    def _make_mask(self, batch_size: int, context_window: int):
        mask = torch.tril(torch.ones((context_window, context_window)))
        return mask.reshape(batch_size, 1, context_window, context_window)


class Config:
    vocab_size: int = 5_000
    context_window: int = 512
    d_model = 768
    num_of_layers: int = 12
    dropout_prob: float = 0.1
    num_of_heads: int = 12
    ffn_hidden_size: int = 3072
    device: str = "cpu"
    n_class: int = 2
    weight_init_params: tuple = (0.0, 0.02)
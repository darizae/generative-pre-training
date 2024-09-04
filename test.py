import torch
from config import Config
from finetuning import CLSHead, LMHead
from gpt_model import GPT

if __name__ == "__main__":

    config = Config()
    gpt = GPT(config)

    lm_test = LMHead(config, gpt)
    cls_test = CLSHead(config, gpt)

    # Generate a random input tensor of shape (batch_size=1, context_window=512)
    # The random integers are sampled between 0 and vocab_size (to simulate token indices)
    logits = lm_test(torch.randint(0, config.vocab_size, (1, config.context_window)))
    
    # The shape should be (batch_size=1, context_window=512, vocab_size=5000)
    # It represents the model's predictions for each token in the sequence (over the entire vocabulary)
    print(f"Language Model Output (lm_test) - Shape: {logits.shape}")

    # Run the classification head on another random input tensor
    lm_logits, cls_logits = cls_test(torch.randint(0, config.vocab_size, (1, config.context_window)))
    
    # Should values:
    # lm_logits: (batch_size=1, context_window=512, vocab_size=5000) - the output for the language modeling task
    # cls_logits: (batch_size=1, context_window=512, n_class=2) - the output for the classification task
    print(f"Classification Logits (cls_test) - Shape: {cls_logits.shape}")
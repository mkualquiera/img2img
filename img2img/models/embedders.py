"""Text and image embedders.
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)

    Taken from https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/
    encoders/modules.py and edited a bit
    """

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()  # type: ignore
        for param in self.parameters():
            param.requires_grad = False

    def tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        return tokens["input_ids"].to(self.device)

    def forward(self, tokens):
        outputs = self.transformer(input_ids=tokens)  # type: ignore

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

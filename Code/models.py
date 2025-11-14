import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
    ViTModel,
    SwinModel
)
import config


class TextEncoder(nn.Module):
    """
    Text encoder supporting BanglaBERT and XLM-RoBERTa
    Returns both CLS token and all hidden states
    """
    def __init__(self, model_name="banglabert"):
        super(TextEncoder, self).__init__()

        self.model_name = model_name
        model_path = config.MODEL_NAMES[model_name]

        print(f"Loading text encoder: {model_path}")
        self.encoder = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Get hidden size from model config
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            cls_token: [batch_size, hidden_size] - CLS token representation
            all_tokens: [batch_size, seq_len, hidden_size] - All token representations
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Last hidden state: [batch_size, seq_len, hidden_size]
        all_tokens = outputs.last_hidden_state

        # CLS token is the first token
        cls_token = all_tokens[:, 0, :]

        return cls_token, all_tokens

    def get_tokenizer(self):
        return self.tokenizer


class ImageEncoder(nn.Module):
    """
    Image encoder supporting ViT and Swin Transformer
    Returns both CLS token and all patch embeddings
    """
    def __init__(self, model_name="swin"):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name
        model_path = config.MODEL_NAMES[model_name]

        print(f"Loading image encoder: {model_path}")

        if model_name == "vit":
            self.encoder = ViTModel.from_pretrained(model_path)
        elif model_name == "swin":
            self.encoder = SwinModel.from_pretrained(model_path)
        else:
            raise ValueError(f"Unknown image encoder: {model_name}")

        self.processor = AutoImageProcessor.from_pretrained(model_path)

        # Get hidden size from model config
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: [batch_size, 3, height, width]

        Returns:
            cls_token: [batch_size, hidden_size] - CLS/pooled representation
            all_tokens: [batch_size, num_patches, hidden_size] - All patch embeddings
        """
        outputs = self.encoder(
            pixel_values=pixel_values,
            return_dict=True
        )

        if self.model_name == "vit":
            # ViT has explicit CLS token
            all_tokens = outputs.last_hidden_state  # [batch_size, num_patches+1, hidden_size]
            cls_token = all_tokens[:, 0, :]  # First token is CLS

        elif self.model_name == "swin":
            # use pooler output
            cls_token = outputs.pooler_output  # [batch_size, hidden_size]

            # For all tokens, we need to use the last hidden state
            # Swin outputs are hierarchical, we use the last layer
            all_tokens = outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]

            # If all_tokens is 4D (hierarchical), flatten spatial dimensions
            if len(all_tokens.shape) == 4:  # [batch_size, height, width, hidden_size]
                batch_size = all_tokens.shape[0]
                all_tokens = all_tokens.view(batch_size, -1, self.hidden_size)

        return cls_token, all_tokens

    def get_processor(self):
        return self.processor


class ProjectionLayer(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(ProjectionLayer, self).__init__()

        if input_size != output_size:
            self.projection = nn.Linear(input_size, output_size)
            self.needs_projection = True
        else:
            self.projection = nn.Identity()
            self.needs_projection = False

    def forward(self, x):
        return self.projection(x)


def create_text_encoder(model_name="banglabert"):
    """Factory function to create text encoder"""
    return TextEncoder(model_name)


def create_image_encoder(model_name="swin"):
    """Factory function to create image encoder"""
    return ImageEncoder(model_name)


if __name__ == "__main__":
    # Test the encoders
    print("Testing Text Encoder...")
    text_encoder = create_text_encoder("banglabert")
    print(f"Text encoder hidden size: {text_encoder.hidden_size}")

    print("\nTesting Image Encoder...")
    image_encoder = create_image_encoder("swin")
    print(f"Image encoder hidden size: {image_encoder.hidden_size}")

    # Test with dummy inputs
    print("\nTesting with dummy inputs...")
    batch_size = 2

    # Text input
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    text_cls, text_all = text_encoder(input_ids, attention_mask)
    print(f"Text CLS shape: {text_cls.shape}")
    print(f"Text all tokens shape: {text_all.shape}")

    # Image input
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    image_cls, image_all = image_encoder(pixel_values)
    print(f"Image CLS shape: {image_cls.shape}")
    print(f"Image all tokens shape: {image_all.shape}")

    print("\n✓ Encoders working correctly!")

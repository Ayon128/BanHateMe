import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config


class HatefulMemeDataset(Dataset):
    """
    Dataset for Bangla Hateful Meme with hierarchical labels

    CSV Structure:
        - image: filename 
        - Text: Bangla text
        - Hate Categories: "Abusive", "Political", etc.
        - Targeted Groups: "Individual", "Community", etc.
        - Label: "Hate" or "Non-Hate"
    """

    # Label mappings
    BINARY_MAP = {"Non-Hate": 0, "Hate": 1}

    CATEGORY_MAP = {
        "Abusive": 0,
        "Political": 1,
        "Gender": 2,
        "Personal Offence": 3,
        "Religious": 4
    }

    TARGET_MAP = {
        "Community": 0,
        "Individual": 1,
        "Organization": 2,
        "Society": 3
    }

    def __init__(self, csv_file, image_dir, text_tokenizer, image_processor, max_length=128):
        """
        Args:
            csv_file: Path to CSV file
            image_dir: Directory containing meme images
            text_tokenizer: Tokenizer for text encoding
            image_processor: Processor for image preprocessing
            max_length: Maximum text length
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        print(f"Loaded {len(self.df)} samples from {csv_file}")

        # Verify image directory exists
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            dict containing:
                - input_ids: Tokenized text
                - attention_mask: Attention mask for text
                - pixel_values: Preprocessed image
                - binary_label: 0 (Non-Hate) or 1 (Hate)
                - category_label: 0-4 (category index)
                - target_label: 0-3 (target group index)
                - text: Original text (for debugging)
                - image_name: Image filename (for debugging)
        """
        row = self.df.iloc[idx]

        # Get text
        text = str(row['Text'])

        # Get image path
        image_name = row['image']
        image_path = os.path.join(self.image_dir, image_name)

        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if error
            image = Image.new('RGB', (224, 224), color='white')

        # Tokenize text
        text_encoding = self.text_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Process image
        image_encoding = self.image_processor(
            images=image,
            return_tensors='pt'
        )

        # Get labels
        binary_label = self.BINARY_MAP.get(row['Label'], 0)
        category_label = self.CATEGORY_MAP.get(row['Hate Categories'], 0)
        target_label = self.TARGET_MAP.get(row['Targeted Groups'], 0)

        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': image_encoding['pixel_values'].squeeze(0),
            'binary_label': torch.tensor(binary_label, dtype=torch.long),
            'category_label': torch.tensor(category_label, dtype=torch.long),
            'target_label': torch.tensor(target_label, dtype=torch.long),
            'text': text,
            'image_name': image_name
        }


def create_dataloaders(text_tokenizer, image_processor, batch_size=16, num_workers=0):
    """
    Creates train, validation, and test dataloaders

    Args:
        text_tokenizer: Tokenizer for text
        image_processor: Processor for images
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = HatefulMemeDataset(
        csv_file=config.TRAIN_CSV,
        image_dir=config.IMAGE_DIR,
        text_tokenizer=text_tokenizer,
        image_processor=image_processor,
        max_length=config.MAX_TEXT_LENGTH
    )

    val_dataset = HatefulMemeDataset(
        csv_file=config.VAL_CSV,
        image_dir=config.IMAGE_DIR,
        text_tokenizer=text_tokenizer,
        image_processor=image_processor,
        max_length=config.MAX_TEXT_LENGTH
    )

    test_dataset = HatefulMemeDataset(
        csv_file=config.TEST_CSV,
        image_dir=config.IMAGE_DIR,
        text_tokenizer=text_tokenizer,
        image_processor=image_processor,
        max_length=config.MAX_TEXT_LENGTH
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def get_label_distributions(csv_file):
    """
    Analyze label distribution in dataset

    Args:
        csv_file: Path to CSV file

    Returns:
        dict with label counts
    """
    df = pd.read_csv(csv_file)

    print(f"\n{'='*50}")
    print(f"Label Distribution in {csv_file}")
    print(f"{'='*50}")

    print(f"\nTotal samples: {len(df)}")

    print(f"\nBinary Labels:")
    print(df['Label'].value_counts())

    print(f"\nHate Categories:")
    print(df['Hate Categories'].value_counts())

    print(f"\nTargeted Groups:")
    print(df['Targeted Groups'].value_counts())

    return {
        'binary': df['Label'].value_counts().to_dict(),
        'category': df['Hate Categories'].value_counts().to_dict(),
        'target': df['Targeted Groups'].value_counts().to_dict()
    }


if __name__ == "__main__":
    # Test dataset loading
    print("Testing Dataset Loading...")

    # Analyze label distributions
    print("\n" + "="*70)
    print("ANALYZING DATASET DISTRIBUTIONS")
    print("="*70)

    get_label_distributions(config.TRAIN_CSV)
    get_label_distributions(config.VAL_CSV)
    get_label_distributions(config.TEST_CSV)

    # Test with actual encoders
    from models import create_text_encoder, create_image_encoder

    print("\n" + "="*70)
    print("TESTING DATALOADER")
    print("="*70)

    text_encoder = create_text_encoder(config.TEXT_ENCODER)
    image_encoder = create_image_encoder(config.IMAGE_ENCODER)

    train_loader, val_loader, test_loader = create_dataloaders(
        text_tokenizer=text_encoder.get_tokenizer(),
        image_processor=image_encoder.get_processor(),
        batch_size=2 

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    print("\nTesting one batch from train loader...")
    batch = next(iter(train_loader))

    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Pixel values shape: {batch['pixel_values'].shape}")
    print(f"Binary labels shape: {batch['binary_label'].shape}")
    print(f"Category labels shape: {batch['category_label'].shape}")
    print(f"Target labels shape: {batch['target_label'].shape}")

    print("\nSample texts:")
    for i, text in enumerate(batch['text'][:2]):
        print(f"  {i+1}. {text[:100]}...")

    print("\nSample images:")
    for i, img_name in enumerate(batch['image_name'][:2]):
        print(f"  {i+1}. {img_name}")

    print("\n✓ Dataset loading working correctly!")

import os
import sys


def test_imports():
    """Test if all required packages can be imported"""
    print("\n" + "="*70)
    print("TESTING IMPORTS")
    print("="*70)

    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('transformers', 'Transformers'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn')
    ]

    all_imports_ok = True

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - MISSING")
            print(f"  Error: {e}")
            all_imports_ok = False

    if all_imports_ok:
        print("\n✓ All required packages are installed!")
        return True
    else:
        print("\n✗ Some packages are missing. Please run:")
        print("   pip install -r requirements.txt")
        return False


def test_device():
    """Test GPU/CPU availability"""
    print("\n" + "="*70)
    print("TESTING DEVICE")
    print("="*70)

    import torch

    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ GPU Not Available - Will use CPU")
        print("  Note: Training will be much slower on CPU")
        return True


def test_data_files():
    """Test if data files exist"""
    print("\n" + "="*70)
    print("TESTING DATA FILES")
    print("="*70)

    import config

    files_to_check = [
        (config.TRAIN_CSV, "Training CSV"),
        (config.VAL_CSV, "Validation CSV"),
        (config.TEST_CSV, "Test CSV"),
        (config.IMAGE_DIR, "Image Directory")
    ]

    all_files_ok = True

    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"✓ {description:20s} - OK ({size:,} bytes)")
            else:
                count = len(os.listdir(filepath))
                print(f"✓ {description:20s} - OK ({count} files)")
        else:
            print(f"✗ {description:20s} - NOT FOUND")
            print(f"  Expected path: {filepath}")
            all_files_ok = False

    if all_files_ok:
        print("\n✓ All data files found!")
        return True
    else:
        print("\n✗ Some data files are missing. Please check paths in config.py")
        return False


def test_csv_structure():
    """Test if CSV files have correct structure"""
    print("\n" + "="*70)
    print("TESTING CSV STRUCTURE")
    print("="*70)

    import pandas as pd
    import config

    required_columns = ['image', 'Text', 'Hate Categories', 'Targeted Groups', 'Label']

    try:
        df = pd.read_csv(config.TRAIN_CSV)

        print(f"Train CSV shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"\n✗ Missing columns: {missing_columns}")
            return False
        else:
            print(f"\n✓ All required columns present!")

            # Check label distributions
            print(f"\nLabel distribution:")
            print(f"  Binary: {df['Label'].value_counts().to_dict()}")
            print(f"  Categories: {df['Hate Categories'].nunique()} unique")
            print(f"  Targets: {df['Targeted Groups'].nunique()} unique")

            return True

    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return False


def test_encoders():
    """Test if encoders can be loaded"""
    print("\n" + "="*70)
    print("TESTING ENCODERS")
    print("="*70)

    try:
        from models import create_text_encoder, create_image_encoder
        import config

        print(f"\nLoading {config.TEXT_ENCODER} text encoder...")
        text_encoder = create_text_encoder(config.TEXT_ENCODER)
        print(f"✓ Text encoder loaded (hidden size: {text_encoder.hidden_size})")

        print(f"\nLoading {config.IMAGE_ENCODER} image encoder...")
        image_encoder = create_image_encoder(config.IMAGE_ENCODER)
        print(f"✓ Image encoder loaded (hidden size: {image_encoder.hidden_size})")

        return True

    except Exception as e:
        print(f"✗ Error loading encoders: {e}")
        print(f"\nNote: First time loading will download models from HuggingFace")
        print(f"      This requires internet connection and may take a few minutes")
        return False


def test_fusion_modules():
    """Test if fusion modules work"""
    print("\n" + "="*70)
    print("TESTING FUSION MODULES")
    print("="*70)

    try:
        from fusion import create_fusion_module
        import torch

        fusion_types = ["summation", "concatenation", "coattention"]

        for fusion_type in fusion_types:
            print(f"\nTesting {fusion_type} fusion...")
            fusion = create_fusion_module(fusion_type, 768)

            # Test forward pass
            batch_size = 2
            text_cls = torch.randn(batch_size, 768)
            image_cls = torch.randn(batch_size, 768)
            text_all = torch.randn(batch_size, 128, 768)
            image_all = torch.randn(batch_size, 49, 768)

            output = fusion(text_cls, image_cls, text_all, image_all)

            if output.shape == (batch_size, 768):
                print(f"✓ {fusion_type} fusion working correctly")
            else:
                print(f"✗ {fusion_type} fusion output shape incorrect: {output.shape}")
                return False

        print(f"\n✓ All fusion modules working!")
        return True

    except Exception as e:
        print(f"✗ Error testing fusion modules: {e}")
        return False


def test_dataset_loading():
    """Test if dataset can be loaded"""
    print("\n" + "="*70)
    print("TESTING DATASET LOADING")
    print("="*70)

    try:
        from models import create_text_encoder, create_image_encoder
        from dataset import create_dataloaders
        import config

        print("Creating encoders...")
        text_encoder = create_text_encoder(config.TEXT_ENCODER)
        image_encoder = create_image_encoder(config.IMAGE_ENCODER)

        print("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            text_tokenizer=text_encoder.get_tokenizer(),
            image_processor=image_encoder.get_processor(),
            batch_size=2,  # Small batch for testing
            num_workers=0
        )

        print(f"\n✓ Dataloaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Test loading one batch
        print("\nLoading one batch from train loader...")
        batch = next(iter(train_loader))

        print(f"✓ Batch loaded successfully!")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        print(f"  Binary labels shape: {batch['binary_label'].shape}")

        return True

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test if complete model can be created"""
    print("\n" + "="*70)
    print("TESTING MODEL CREATION")
    print("="*70)

    try:
        from train import HierarchicalMultimodalModel
        import config
        import torch

        print("Creating hierarchical multimodal model...")
        model = HierarchicalMultimodalModel(
            config.TEXT_ENCODER,
            config.IMAGE_ENCODER,
            config.FUSION_TYPE
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n✓ Model created successfully!")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        input_ids = torch.randint(0, 1000, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128)
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        model.eval()
        with torch.no_grad():
            binary_logits, category_logits, target_logits = model(
                input_ids, attention_mask, pixel_values
            )

        print(f"✓ Forward pass successful!")
        print(f"  Binary logits shape: {binary_logits.shape}")
        print(f"  Category logits shape: {category_logits.shape}")
        print(f"  Target logits shape: {target_logits.shape}")

        return True

    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("SETUP VERIFICATION TEST")
    print("="*70)
    print("This script will verify your setup before training")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("Device (GPU/CPU)", test_device),
        ("Data Files", test_data_files),
        ("CSV Structure", test_csv_structure),
        ("Encoders", test_encoders),
        ("Fusion Modules", test_fusion_modules),
        ("Dataset Loading", test_dataset_loading),
        ("Model Creation", test_model_creation)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed with error: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30s} - {status}")

    print("\n" + "-"*70)
    print(f"Tests passed: {passed}/{total}")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests passed! You're ready to start training!")
        print("\nRun: python train.py")
        return True
    else:
        print("\n⚠ Some tests failed. Please fix the issues above before training.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

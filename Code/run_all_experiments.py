import os
import json
import subprocess
import time
from datetime import datetime


# All model combinations 
TEXT_ENCODERS = ["banglabert", "xlm-roberta"]
IMAGE_ENCODERS = ["swin", "vit"]
FUSION_TYPES = ["summation", "concatenation", "coattention"]

# Alpha/Beta combinations for ablation study
ALPHA_BETA_PAIRS = [
    (0.2, 0.8),
    (0.5, 0.5),
    (0.8, 0.2),
    (1.0, 1.0)
]


def update_config(text_encoder, image_encoder, fusion_type, alpha, beta):
    """
    Update config.py with new experiment settings

    Args:
        text_encoder: Text encoder name
        image_encoder: Image encoder name
        fusion_type: Fusion strategy
        alpha: Alpha value
        beta: Beta value
    """
    config_path = "config.py"

    # Read current config
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Update relevant lines
    updated_lines = []
    for line in lines:
        if line.strip().startswith('TEXT_ENCODER ='):
            updated_lines.append(f'TEXT_ENCODER = "{text_encoder}"\n')
        elif line.strip().startswith('IMAGE_ENCODER ='):
            updated_lines.append(f'IMAGE_ENCODER = "{image_encoder}"\n')
        elif line.strip().startswith('FUSION_TYPE ='):
            updated_lines.append(f'FUSION_TYPE = "{fusion_type}"\n')
        elif line.strip().startswith('ALPHA ='):
            updated_lines.append(f'ALPHA = {alpha}\n')
        elif line.strip().startswith('BETA ='):
            updated_lines.append(f'BETA = {beta}\n')
        else:
            updated_lines.append(line)

    # Write updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)


def run_experiment(text_encoder, image_encoder, fusion_type, alpha, beta, log_dir):
    """
    Run a single experiment

    Args:
        text_encoder: Text encoder name
        image_encoder: Image encoder name
        fusion_type: Fusion strategy
        alpha: Alpha value
        beta: Beta value
        log_dir: Directory to save logs

    Returns:
        dict with experiment results
    """
    exp_name = f"{text_encoder}_{image_encoder}_{fusion_type}_a{alpha}_b{beta}"

    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("="*80)
    print(f"Text Encoder: {text_encoder}")
    print(f"Image Encoder: {image_encoder}")
    print(f"Fusion Type: {fusion_type}")
    print(f"Alpha: {alpha}, Beta: {beta}")
    print("="*80 + "\n")

    # Update config
    update_config(text_encoder, image_encoder, fusion_type, alpha, beta)

    # Run training
    start_time = time.time()

    try:
        # Run train.py and capture output
        result = subprocess.run(
            ['python', 'train.py'],
            capture_output=True,
            text=True,
            timeout=None  
        )

        elapsed_time = time.time() - start_time

        # Save logs
        log_file = os.path.join(log_dir, f"{exp_name}.log")
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Start time: {datetime.now()}\n")
            f.write(f"Duration: {elapsed_time:.2f} seconds\n")
            f.write("\n" + "="*80 + "\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n" + "="*80 + "\n")
            f.write("STDERR:\n")
            f.write(result.stderr)

        # Check if successful
        if result.returncode == 0:
            print(f"\n✓ Experiment completed successfully in {elapsed_time:.2f} seconds")

            # Try to load results
            results_path = f"output/{exp_name}/results.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                return {
                    'status': 'success',
                    'time': elapsed_time,
                    'results': results
                }
            else:
                return {
                    'status': 'success_no_results',
                    'time': elapsed_time
                }
        else:
            print(f"\n✗ Experiment failed with return code {result.returncode}")
            return {
                'status': 'failed',
                'time': elapsed_time,
                'error': result.stderr
            }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Experiment failed with exception: {str(e)}")
        return {
            'status': 'error',
            'time': elapsed_time,
            'error': str(e)
        }


def run_all_base_experiments():
    """
    Run all 12 base model combinations (4 encoders × 3 fusions)
    Uses default alpha=0.5, beta=0.5 for initial comparison
    """
    print("\n" + "="*80)
    print("RUNNING ALL BASE MODEL COMBINATIONS")
    print("="*80)
    print(f"Total experiments: {len(TEXT_ENCODERS) * len(IMAGE_ENCODERS) * len(FUSION_TYPES)}")
    print("="*80 + "\n")

    # Create log directory
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)

    all_results = {}
    alpha, beta = 0.5, 0.5  # Default values

    exp_count = 0
    total_experiments = len(TEXT_ENCODERS) * len(IMAGE_ENCODERS) * len(FUSION_TYPES)

    for text_encoder in TEXT_ENCODERS:
        for image_encoder in IMAGE_ENCODERS:
            for fusion_type in FUSION_TYPES:
                exp_count += 1

                print(f"\n{'='*80}")
                print(f"EXPERIMENT {exp_count}/{total_experiments}")
                print(f"{'='*80}")

                exp_name = f"{text_encoder}_{image_encoder}_{fusion_type}_a{alpha}_b{beta}"

                result = run_experiment(
                    text_encoder, image_encoder, fusion_type,
                    alpha, beta, log_dir
                )

                all_results[exp_name] = result

                # Save intermediate results
                summary_path = os.path.join(log_dir, "experiment_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ALL BASE EXPERIMENTS COMPLETED")
    print("="*80)

    return all_results


def run_alpha_beta_ablation(best_model_config):
    """
    Run alpha/beta ablation study on the best model

    Args:
        best_model_config: Dictionary with text_encoder, image_encoder, fusion_type
    """
    print("\n" + "="*80)
    print("RUNNING ALPHA/BETA ABLATION STUDY")
    print("="*80)
    print(f"Model: {best_model_config['text_encoder']} + "
          f"{best_model_config['image_encoder']} + "
          f"{best_model_config['fusion_type']}")
    print(f"Total experiments: {len(ALPHA_BETA_PAIRS)}")
    print("="*80 + "\n")

    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)

    all_results = {}

    for idx, (alpha, beta) in enumerate(ALPHA_BETA_PAIRS, 1):
        print(f"\n{'='*80}")
        print(f"ABLATION EXPERIMENT {idx}/{len(ALPHA_BETA_PAIRS)}")
        print(f"Alpha: {alpha}, Beta: {beta}")
        print(f"{'='*80}")

        exp_name = (f"{best_model_config['text_encoder']}_"
                   f"{best_model_config['image_encoder']}_"
                   f"{best_model_config['fusion_type']}_a{alpha}_b{beta}")

        result = run_experiment(
            best_model_config['text_encoder'],
            best_model_config['image_encoder'],
            best_model_config['fusion_type'],
            alpha, beta, log_dir
        )

        all_results[exp_name] = result

        # Save intermediate results
        summary_path = os.path.join(log_dir, "ablation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ALPHA/BETA ABLATION COMPLETED")
    print("="*80)

    return all_results


def print_summary(results):
    """Print experiment summary"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    successful = 0
    failed = 0
    total_time = 0

    for exp_name, result in results.items():
        if result['status'] == 'success':
            successful += 1
            total_time += result['time']

            if 'results' in result:
                test_metrics = result['results']['test_metrics']
                binary_f1 = test_metrics['binary']['f1']
                print(f"\n{exp_name}:")
                print(f"  Status: ✓ Success")
                print(f"  Time: {result['time']:.2f}s")
                print(f"  Binary F1: {binary_f1:.4f}")
        else:
            failed += 1
            print(f"\n{exp_name}:")
            print(f"  Status: ✗ Failed")

    print("\n" + "="*80)
    print(f"Total Experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPLETE PAPER REPLICATION - EXPERIMENT RUNNER")
    print("="*80)
    
    print("\nSelect experiment mode:")
    print("1. Run all 12 base model combinations (α=0.5, β=0.5)")
    print("2. Run alpha/beta ablation on specific model")
    print("3. Run complete replication (base + ablation)")
    print("4. Run single experiment with custom config")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        # Run all base experiments
        results = run_all_base_experiments()
        print_summary(results)

    elif choice == "2":
        # Run alpha/beta ablation
        print("\nEnter best model configuration:")
        text_encoder = input("Text encoder (banglabert/xlm-roberta): ").strip()
        image_encoder = input("Image encoder (swin/vit): ").strip()
        fusion_type = input("Fusion type (summation/concatenation/coattention): ").strip()

        best_model = {
            'text_encoder': text_encoder,
            'image_encoder': image_encoder,
            'fusion_type': fusion_type
        }

        results = run_alpha_beta_ablation(best_model)
        print_summary(results)

    elif choice == "3":
        # Run complete 
        print("\nRunning complete...")
        confirm = input("Continue? (yes/no): ").strip().lower()

        if confirm == 'yes':
            # Run base experiments
            base_results = run_all_base_experiments()

            # Find best model from base experiments
            best_f1 = 0
            best_model = None

            for exp_name, result in base_results.items():
                if result['status'] == 'success' and 'results' in result:
                    f1 = result['results']['test_metrics']['binary']['f1']
                    if f1 > best_f1:
                        best_f1 = f1
                        
                        parts = exp_name.split('_')
                        best_model = {
                            'text_encoder': parts[0],
                            'image_encoder': parts[1],
                            'fusion_type': parts[2]
                        }

            print(f"\nBest model found: {best_model} with F1={best_f1:.4f}")

            # Run ablation on best model
            if best_model:
                ablation_results = run_alpha_beta_ablation(best_model)
                all_results = {**base_results, **ablation_results}
                print_summary(all_results)

    elif choice == "4":
        # Run single experiment
        print("\nEnter experiment configuration:")
        text_encoder = input("Text encoder (banglabert/xlm-roberta): ").strip()
        image_encoder = input("Image encoder (swin/vit): ").strip()
        fusion_type = input("Fusion type (summation/concatenation/coattention): ").strip()
        alpha = float(input("Alpha value: ").strip())
        beta = float(input("Beta value: ").strip())

        log_dir = "experiment_logs"
        os.makedirs(log_dir, exist_ok=True)

        result = run_experiment(text_encoder, image_encoder, fusion_type, alpha, beta, log_dir)

        if result['status'] == 'success':
            print(f"\n✓ Experiment completed successfully!")
            if 'results' in result:
                metrics = result['results']['test_metrics']
                print(f"\nResults:")
                print(f"  Binary F1: {metrics['binary']['f1']:.4f}")
                print(f"  Category F1: {metrics['category']['f1']:.4f}")
                print(f"  Target F1: {metrics['target']['f1']:.4f}")
        else:
            print(f"\n✗ Experiment failed!")

    else:
        print("Invalid choice!")

    print("\n" + "="*80)
    print("EXPERIMENT RUNNER FINISHED")
    print("="*80)

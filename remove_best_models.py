import os
import sys

# Base directory path
BASE_DIR = "/media/16TB_Storage/kavin/amd_siglip/saved_models/optuna_run_20250826_211533"

def remove_best_models():
    """Remove best_model.pt files from trial_0 to trial_499 folders."""
    
    deleted_count = 0
    not_found_count = 0
    error_count = 0
    
    print("Starting to remove best_model.pt files from trial_0 to trial_499...")
    print(f"Base directory: {BASE_DIR}")
    print()
    
    # Check if base directory exists
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory does not exist: {BASE_DIR}")
        return
    
    # /media/16TB_Storage/kavin/amd_siglip/saved_models/optuna_run_20250826_211533/trial_0/best_model.pt

    # Loop through trial_0 to trial_499
    for i in range(500):
        trial_dir = os.path.join(BASE_DIR, f"trial_{i}")
        model_file = os.path.join(trial_dir, "best_model.pt")
        
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
                print(f"Deleted: trial_{i}/best_model.pt")
                deleted_count += 1
            except OSError as e:
                print(f"Failed to delete trial_{i}/best_model.pt: {e}")
                error_count += 1
        else:
            print(f"Not found: trial_{i}/best_model.pt")
            not_found_count += 1
    
    print()
    print("Summary:")
    print(f"Files deleted: {deleted_count}")
    print(f"Files not found: {not_found_count}")
    print(f"Deletion errors: {error_count}")
    print(f"Total processed: 500")

if __name__ == "__main__":
    # Ask for confirmation before proceeding
    response = input("This will delete best_model.pt files from 500 trial folders. Continue? (y/N): ")
    if response.lower() in ['y', 'yes']:
        remove_best_models()
    else:
        print("Operation cancelled.")
        sys.exit(0)
# TRAINING SCRIPT WITH OPTUNA HYPERPARAMETER SEARCH

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor, SiglipVisionModel
import numpy as np
from datetime import datetime
from tfrecord.torch.dataset import TFRecordDataset
from embedder import TextEmbedder
from alignment import SigLIPLoss
import optuna

# Keep same paths as original - just add optuna subdirectory
TFRECORD_PATH = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/data_v4.tfrecord"
BASE_MODEL_SAVE_DIR = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/saved_models"
MODEL_SAVE_DIR = os.path.join(BASE_MODEL_SAVE_DIR, "optuna_search")

# Fixed parameters
MAX_TEXT_LEN = 128
IMAGE_HEIGHT = 1055
IMAGE_WIDTH = 703
IMAGE_CHANNELS = 3

# Device setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

description = {
    "input_ids": "int",
    "attn_mask": "int", 
    "class": "byte", 
    "normalized_image": "byte"
}

def collate_fn(batch):
    """Custom collate function for TFRecord data"""
    images, labels, input_ids_list, attention_masks = [], [], [], []
    
    for item in batch:
        try:
            img_bytes = item["normalized_image"]
            img_array = np.frombuffer(img_bytes, dtype=np.float32)
            # Reshape to (WIDTH, HEIGHT, CHANNELS) as requested
            img = img_array.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS).copy()  # Copy to make writable
            
            # Convert to tensor and resize to 224x224
            img = torch.from_numpy(img).permute(2, 0, 1)  # CHW format
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            images.append(img)
            
            # Decode label
            class_str = item["class"].decode('utf-8')
            label = 0 if class_str == 'n' else 1  # n=normal=0, w=wet=1
            labels.append(label)
            
            # Get input_ids (already tokenized)
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            input_ids_list.append(input_ids)
            
            # Get attention mask from tfrecord
            attention_mask = torch.tensor(item["attn_mask"], dtype=torch.bool)
            attention_masks.append(attention_mask)
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    if len(images) == 0:
        raise RuntimeError("No valid images in batch! Check your data.")
    
    # Stack everything
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_masks)
    
    return images, labels, input_ids, attention_mask

class SigLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.cls_head = nn.Linear(768, 2)

        self.siglip_loss = SigLIPLoss(
            latent_dim=768,
            text_model="google-t5/t5-base",
            max_txt_len=MAX_TEXT_LEN,
            pool="mean",
            dtype=torch.float32
        )
        
    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_encoder(pixel_values=images).last_hidden_state  # (B, N, 768)
        cls_logits = self.cls_head(img_features[:, 0])  # (B, 2)
        align_loss, _, _ = self.siglip_loss(img_features, input_ids, attention_mask)
        return cls_logits, align_loss

def train_epoch(model, dataloader, optimizer, alpha):
    model.train()
    total_loss = total_cls = total_align = 0
    num_batches = 0
    
    for batch_idx, (images, labels, input_ids, attention_mask) in enumerate(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        # Forward pass
        cls_logits, align_loss = model(images, input_ids, attention_mask)
        
        # Losses with hyperparameter alpha
        cls_loss = F.cross_entropy(cls_logits, labels)
        total_loss_batch = alpha * cls_loss + (1 - alpha) * align_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # Accumulate
        total_loss += total_loss_batch.item()
        total_cls += cls_loss.item()
        total_align += align_loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return 0, 0, 0
    
    return total_loss/num_batches, total_cls/num_batches, total_align/num_batches

def create_optimizer(model, optimizer_name, learning_rate, weight_decay):
    """Create optimizer based on trial suggestion"""
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def objective(trial):
    """Optuna objective function to minimize"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    alpha = trial.suggest_float('alpha', 0.1, 0.9)  # Your key loss weighting parameter
    optimizer_name = trial.suggest_categorical('optimizer', ['adamw', 'adam', 'sgd'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Learning rate scheduler parameters
    scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.8)
    scheduler_patience = trial.suggest_int('scheduler_patience', 2, 5)
    
    # Create timestamp for this trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_id = f"trial_{trial.number:03d}_{timestamp}"
    
    # Create trial-specific directory within optuna_search
    trial_save_dir = os.path.join(MODEL_SAVE_DIR, trial_id)
    os.makedirs(trial_save_dir, exist_ok=True)
    
    print(f"\n=== Trial {trial.number} ===")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Alpha: {alpha}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Weight decay: {weight_decay}")
    print(f"Scheduler factor: {scheduler_factor}")
    print(f"Scheduler patience: {scheduler_patience}")
    
    try:
        # Create dataset and dataloader with trial-specific batch size
        dataset = TFRecordDataset(TFRECORD_PATH, None, description)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        
        # Create model
        model = SigLIPModel().to(DEVICE)
        
        # Create optimizer with trial-specific parameters
        optimizer = create_optimizer(model, optimizer_name, learning_rate, weight_decay)
        
        # Learning rate scheduler with trial-specific parameters
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        
        # TensorBoard for this trial (within runs directory)
        writer = SummaryWriter(f'runs/optuna_trial_{trial.number:03d}_{timestamp}')
        
        # Training loop (reduced epochs for faster hyperparameter search)
        max_epochs = 25  # Reduced for faster search
        best_loss = float('inf')
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Train
            avg_total, avg_cls, avg_align = train_epoch(model, dataloader, optimizer, alpha)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Total', avg_total, epoch)
            writer.add_scalar('Loss/Classification', avg_cls, epoch)
            writer.add_scalar('Loss/Alignment', avg_align, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Step scheduler
            scheduler.step(avg_total)
            
            # Track best loss
            if avg_total < best_loss:
                best_loss = avg_total
                patience_counter = 0
                
                # Save best model for this trial
                torch.save({
                    'trial_number': trial.number,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_total,
                    'image_encoder_state_dict': model.image_encoder.state_dict(),
                    'siglip_loss_state_dict': model.siglip_loss.state_dict(),
                    'hyperparameters': {
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'alpha': alpha,
                        'optimizer': optimizer_name,
                        'weight_decay': weight_decay,
                        'scheduler_factor': scheduler_factor,
                        'scheduler_patience': scheduler_patience
                    }
                }, os.path.join(trial_save_dir, 'best_model.pt'))
            else:
                patience_counter += 1
            
            # Report intermediate value to Optuna for pruning
            trial.report(avg_total, epoch)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Trial {trial.number}: Early stopping at epoch {epoch+1}")
                break
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                writer.close()
                raise optuna.exceptions.TrialPruned()
        
        writer.close()
        
        # Save trial results
        trial_results = {
            'trial_number': trial.number,
            'best_loss': best_loss,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'alpha': alpha,
                'optimizer': optimizer_name,
                'weight_decay': weight_decay,
                'scheduler_factor': scheduler_factor,
                'scheduler_patience': scheduler_patience
            }
        }
        
        with open(os.path.join(trial_save_dir, 'trial_results.json'), 'w') as f:
            json.dump(trial_results, f, indent=2)
        
        print(f"Trial {trial.number} completed: Best loss = {best_loss:.4f}")
        return best_loss
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise e

def run_hyperparameter_optimization():
    """Run the hyperparameter optimization study"""
    
    # Create study directory
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # Create study
    study_name = f"siglip_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # We want to minimize loss
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"Starting SigLIP Hyperparameter Optimization")
    print("=" * 50)
    print(f"Study name: {study_name}")
    print(f"Results will be saved to: {MODEL_SAVE_DIR}")
    print(f"TFRecord path: {TFRECORD_PATH}")
    print(f"Device: {DEVICE}")
    
    # Optimize
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (loss): {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save study results in the main optuna_search directory
    study_results = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'optimization_completed_at': datetime.now().isoformat(),
        'tfrecord_path': TFRECORD_PATH,
        'device_used': DEVICE,
        'all_trials': []
    }
    
    # Save all trial results
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': str(trial.state)
        }
        study_results['all_trials'].append(trial_info)
    
    # Save study results to main directory
    with open(os.path.join(MODEL_SAVE_DIR, 'study_results.json'), 'w') as f:
        json.dump(study_results, f, indent=2)
    
    # Copy best model to main directory for easy access
    best_trial_dir = os.path.join(MODEL_SAVE_DIR, f"trial_{study.best_trial.number:03d}_*")
    import glob
    best_trial_dirs = glob.glob(best_trial_dir)
    if best_trial_dirs:
        best_model_path = os.path.join(best_trial_dirs[0], 'best_model.pt')
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, os.path.join(MODEL_SAVE_DIR, 'best_model_overall.pt'))
    
    # Generate optimization history plot
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(MODEL_SAVE_DIR, 'optimization_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot parameter importances
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(os.path.join(MODEL_SAVE_DIR, 'param_importances.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization plots saved to {MODEL_SAVE_DIR}")
    except ImportError:
        print("\nInstall matplotlib for visualization: pip install matplotlib")
    
    print(f"\nOptimization completed!")
    print(f"Best hyperparameters saved to: {os.path.join(MODEL_SAVE_DIR, 'study_results.json')}")
    print(f"Use these parameters in your main training script.")
    
    return study

if __name__ == "__main__":
    print("SigLIP Hyperparameter Optimization with Optuna")
    print("Using your existing directory structure with optuna_search subdirectory")
    print("=" * 70)
    
    study = run_hyperparameter_optimization()
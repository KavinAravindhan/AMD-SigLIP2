# THIS VERSION LOADS BEST HYPERPARAMETERS FROM OPTUNA SEARCH

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

# Keep same paths as original
TFRECORD_PATH = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/data_v4.tfrecord"
BASE_MODEL_SAVE_DIR = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/saved_models"
OPTUNA_RESULTS_PATH = os.path.join(BASE_MODEL_SAVE_DIR, "optuna_search", "study_results.json")

# Fixed parameters
MAX_TEXT_LEN = 128
IMAGE_HEIGHT = 1055
IMAGE_WIDTH = 703
IMAGE_CHANNELS = 3

# FULL TRAINING EPOCHS (not reduced like in Optuna search)
FULL_EPOCHS = 100

# Device setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

def load_best_hyperparameters():
    """Load the best hyperparameters from Optuna study results"""
    if not os.path.exists(OPTUNA_RESULTS_PATH):
        print(f"Optuna results not found at: {OPTUNA_RESULTS_PATH}")
        print("Please run train_optuna.py first to generate optimal hyperparameters.")
        print("\nUsing default hyperparameters...")
        return {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'alpha': 0.5,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'scheduler_factor': 0.5,
            'scheduler_patience': 3
        }, False
    
    with open(OPTUNA_RESULTS_PATH, 'r') as f:
        study_results = json.load(f)
    
    best_params = study_results['best_params']
    print(f"Loaded BEST hyperparameters from Optuna (Trial {study_results['best_trial_number']}):")
    print(f"Best loss achieved during search: {study_results['best_value']:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()
    
    return best_params, True

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

def train_epoch(model, dataloader, optimizer, alpha, epoch):
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
        
        # Losses with optimized alpha
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
        
        # Print progress less frequently for full training
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Total={total_loss_batch.item():.4f}, Cls={cls_loss.item():.4f}, Align={align_loss.item():.4f}")
    
    if num_batches == 0:
        return 0, 0, 0
    
    return total_loss/num_batches, total_cls/num_batches, total_align/num_batches

def create_optimizer(model, optimizer_name, learning_rate, weight_decay):
    """Create optimizer based on best parameters"""
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def save_model(model, optimizer, scheduler, epoch, loss, filename, run_save_dir, best_params, used_optuna):
    filepath = os.path.join(run_save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'image_encoder_state_dict': model.image_encoder.state_dict(),
        'siglip_loss_state_dict': model.siglip_loss.state_dict(),
        'hyperparameters_source': 'optuna_optimized' if used_optuna else 'default',
        'best_hyperparameters': best_params,
    }, filepath)
    print(f"Saved: {filepath}")

def main():
    # Load best hyperparameters from Optuna
    best_params, used_optuna = load_best_hyperparameters()
    
    # Extract hyperparameters
    LEARNING_RATE = best_params['learning_rate']
    BATCH_SIZE = best_params['batch_size']
    ALPHA = best_params['alpha']
    OPTIMIZER_NAME = best_params['optimizer']
    WEIGHT_DECAY = best_params['weight_decay']
    SCHEDULER_FACTOR = best_params['scheduler_factor']
    SCHEDULER_PATIENCE = best_params['scheduler_patience']
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_type = "optuna_optimized" if used_optuna else "default_params"
    run_save_dir = os.path.join(BASE_MODEL_SAVE_DIR, f"{run_type}_{timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)
    
    print(f"Starting SigLIP training with {'OPTUNA-OPTIMIZED' if used_optuna else 'DEFAULT'} hyperparameters...")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Alpha (loss weighting): {ALPHA}")
    print(f"Optimizer: {OPTIMIZER_NAME}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Scheduler factor: {SCHEDULER_FACTOR}")
    print(f"Scheduler patience: {SCHEDULER_PATIENCE}")
    print(f"Epochs: {FULL_EPOCHS}")
    print(f"Model save directory: {run_save_dir}\n")
    
    # Save training config with best hyperparameters
    config = {
        'timestamp': timestamp,
        'device': DEVICE,
        'epochs': FULL_EPOCHS,
        'max_text_len': MAX_TEXT_LEN,
        'image_dimensions': f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}",
        'tfrecord_path': TFRECORD_PATH,
        'hyperparameters_source': 'optuna_optimized' if used_optuna else 'default',
        'hyperparameters': best_params,
        'optuna_results_path': OPTUNA_RESULTS_PATH if used_optuna else None
    }
    
    with open(os.path.join(run_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataset and dataloader with optimized batch size
    dataset = TFRecordDataset(TFRECORD_PATH, None, description)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    
    # Dataset info (same as original)
    print("Checking dataset...")
    test_dataset = TFRecordDataset(TFRECORD_PATH, None, description)
    
    sample_count = 0
    first_sample_info = None
    
    for i, item in enumerate(test_dataset):
        sample_count += 1
        
        if i == 0:
            img_bytes = item["normalized_image"]
            first_sample_info = {
                'bytes': len(img_bytes),
                'expected_bytes': IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS*4,
                'class': item['class'].decode('utf-8'),
                'input_ids_len': len(item['input_ids']),
                'attn_mask_len': len(item['attn_mask'])
            }
    
    print(f"Total samples in dataset: {sample_count}")
    print(f"Expected number of batches: {(sample_count + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"First sample: {first_sample_info['bytes']} bytes (expected: {first_sample_info['expected_bytes']} bytes)")
    print(f"Class: {first_sample_info['class']}")
    print(f"Input IDs length: {first_sample_info['input_ids_len']}")
    print(f"Attention mask length: {first_sample_info['attn_mask_len']}\n")
    
    # Create model
    model = SigLIPModel().to(DEVICE)
    
    # Create optimizer with optimized parameters
    optimizer = create_optimizer(model, OPTIMIZER_NAME, LEARNING_RATE, WEIGHT_DECAY)
    
    # Learning rate scheduler with optimized parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    
    # TensorBoard
    tb_name = f"siglip_{'optuna_best' if used_optuna else 'default'}_{timestamp}"
    writer = SummaryWriter(f'runs/{tb_name}')
    
    print(f"Starting training for {FULL_EPOCHS} epochs...")
    print(f"Loss formula: total_loss = {ALPHA} * cls_loss + {1-ALPHA} * align_loss")
    if used_optuna:
        print("🎯 Using Optuna-optimized hyperparameters for potentially better performance!")
    else:
        print("⚠️  Using default hyperparameters - consider running train_optuna.py first")
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    history = {'total_loss': [], 'cls_loss': [], 'align_loss': []}
    
    final_epoch = 0
    for epoch in range(FULL_EPOCHS):
        final_epoch = epoch + 1
        print(f"\n=== Epoch {epoch+1}/{FULL_EPOCHS} ===")
        
        # Train
        avg_total, avg_cls, avg_align = train_epoch(model, dataloader, optimizer, ALPHA, epoch+1)
        
        # Save history
        history['total_loss'].append(avg_total)
        history['cls_loss'].append(avg_cls)
        history['align_loss'].append(avg_align)
        
        # Log
        print(f"Total Loss: {avg_total:.4f}, Cls Loss: {avg_cls:.4f}, Align Loss: {avg_align:.4f}")
        writer.add_scalar('Loss/Total', avg_total, epoch)
        writer.add_scalar('Loss/Classification', avg_cls, epoch)
        writer.add_scalar('Loss/Alignment', avg_align, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Step scheduler
        scheduler.step(avg_total)
        
        # Save best model
        if avg_total < best_loss and avg_total > 0:
            best_loss = avg_total
            save_model(model, optimizer, scheduler, epoch+1, avg_total, 'best_model.pt', run_save_dir, best_params, used_optuna)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_model(model, optimizer, scheduler, epoch+1, avg_total, f'checkpoint_epoch_{epoch+1}.pt', run_save_dir, best_params, used_optuna)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    save_model(model, optimizer, scheduler, final_epoch, avg_total, 'final_model.pt', run_save_dir, best_params, used_optuna)
    
    # Save training history
    with open(os.path.join(run_save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Models saved in: {run_save_dir}")
    print(f"Hyperparameters used: {best_params}")
    if used_optuna:
        print("🎉 Training completed with Optuna-optimized hyperparameters!")
    
    writer.close()

if __name__ == "__main__":
    main()
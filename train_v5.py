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

TFRECORD_PATH = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/data_v4.tfrecord"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
MAX_TEXT_LEN = 128  # From the debug output, input_ids have length 128
ALPHA = 0.5  # Loss weighting parameter: total_loss = alpha * cls_loss + (1-alpha) * align_loss

# Model save directory
MODEL_SAVE_DIR = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/saved_models"

# Device setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Create timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create model save directory with timestamp
run_save_dir = os.path.join(MODEL_SAVE_DIR, f"run_{timestamp}")
os.makedirs(run_save_dir, exist_ok=True)
print(f"Models will be saved to: {run_save_dir}")

# TensorBoard
writer = SummaryWriter(f'runs/siglip_training_{timestamp}')

IMAGE_HEIGHT = 703
IMAGE_WIDTH = 1055
IMAGE_CHANNELS = 3

description = {
    "input_ids": "int",
    "attn_mask": "int", 
    "class": "byte", 
    "normalized_image": "byte"  # Contains 1055×703×3 float32 values as bytes
}

def collate_fn(batch):
    """Custom collate function for TFRecord data"""
    images, labels, input_ids_list, attention_masks = [], [], [], []
    
    for item in batch:
        try:
            img_bytes = item["normalized_image"]
            img_array = np.frombuffer(img_bytes, dtype=np.float32)
            img = img_array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).copy()  # Copy to make writable
            
            # Convert to tensor and resize to 384x384 (to match siglip-so400m-patch14-384)
            img = torch.from_numpy(img).permute(2, 0, 1)  # CHW format
            img = F.interpolate(img.unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False).squeeze(0)
            
            # Normalize with SigLIP's expected values: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            # This transforms from [0,1] to [-1,1] range
            img = (img - 0.5) / 0.5
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

dataset = TFRecordDataset(TFRECORD_PATH, None, description)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

class SigLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use Gemma3's SigLIP encoder with 1152 dimensions
        self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        # Update cls_head to use 1152 dimensions instead of 768
        self.cls_head = nn.Linear(1152, 2)

        self.siglip_loss = SigLIPLoss(
            latent_dim=1152,  # Updated to match Gemma3's SigLIP encoder
            text_model="google-t5/t5-base",
            max_txt_len=MAX_TEXT_LEN,  # Now 128
            pool="mean",
            dtype=torch.float32
        )
        
    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_encoder(pixel_values=images).last_hidden_state  # (B, 729, 1152) for 384x384 images
        cls_logits = self.cls_head(img_features[:, 0])  # (B, 2)
        align_loss, _, _ = self.siglip_loss(img_features, input_ids, attention_mask)
        return cls_logits, align_loss

def train_epoch(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = total_cls = total_align = 0
    num_batches = 0
    
    print(f"Starting epoch {epoch}")
    
    for batch_idx, (images, labels, input_ids, attention_mask) in enumerate(dataloader):
        print(f"Processing batch {batch_idx}, batch size: {len(images)}")
        
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        # Forward pass
        cls_logits, align_loss = model(images, input_ids, attention_mask)
        
        # Losses
        cls_loss = F.cross_entropy(cls_logits, labels)
        total_loss_batch = ALPHA * cls_loss + (1 - ALPHA) * align_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # Accumulate
        total_loss += total_loss_batch.item()
        total_cls += cls_loss.item()
        total_align += align_loss.item()
        num_batches += 1
        
        print(f"Epoch {epoch}, Batch {batch_idx}: Total={total_loss_batch.item():.4f}, Cls={cls_loss.item():.4f}, Align={align_loss.item():.4f}")
    
    print(f"Epoch {epoch} completed: Processed {num_batches} batches")
    
    if num_batches == 0:
        print("No valid batches in epoch!")
        return 0, 0, 0
    
    return total_loss/num_batches, total_cls/num_batches, total_align/num_batches

def save_model(model, optimizer, scheduler, epoch, loss, filename):
    # Save in the run-specific directory
    filepath = os.path.join(run_save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'image_encoder_state_dict': model.image_encoder.state_dict(),
        'siglip_loss_state_dict': model.siglip_loss.state_dict(),
        'alpha': ALPHA,
    }, filepath)
    print(f"Saved: {filepath}")

def main():
    print(f"Starting SigLIP training...")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Alpha (loss weighting): {ALPHA}")
    print(f"Using Gemma3 SigLIP encoder with 1152 dimensions (384x384 input)")
    print(f"Model save directory: {run_save_dir}\n")
    
    # Save training config
    config = {
        'timestamp': timestamp,
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'alpha': ALPHA,
        'max_text_len': MAX_TEXT_LEN,
        'image_dimensions': f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}",
        'tfrecord_path': TFRECORD_PATH,
        'siglip_model': 'google/siglip-so400m-patch14-384',
        'input_image_size': '384x384',
        'image_normalization': 'mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)',
        'latent_dim': 1152
    }
    
    import json
    with open(os.path.join(run_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    model = SigLIPModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print("Checking dataset...")
    test_dataset = TFRecordDataset(TFRECORD_PATH, None, description)
    
    # Count ALL samples in dataset
    sample_count = 0
    first_sample_info = None
    
    for i, item in enumerate(test_dataset):
        sample_count += 1
        
        # Save info from first sample
        if i == 0:
            img_bytes = item["normalized_image"]
            first_sample_info = {
                'bytes': len(img_bytes),
                'expected_bytes': IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS*4,  # Total elements * 4 bytes per float32
                'class': item['class'].decode('utf-8'),
                'input_ids_len': len(item['input_ids']),
                'attn_mask_len': len(item['attn_mask'])
            }
    
    # Print dataset statistics
    print(f"Total samples in dataset: {sample_count}")
    print(f"Expected number of batches: {(sample_count + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"First sample: {first_sample_info['bytes']} bytes (expected: {first_sample_info['expected_bytes']} bytes)")
    print(f"Class: {first_sample_info['class']}")
    print(f"Input IDs length: {first_sample_info['input_ids_len']}")
    print(f"Attention mask length: {first_sample_info['attn_mask_len']}\n")
    
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Loss formula: total_loss = {ALPHA} * cls_loss + {1-ALPHA} * align_loss")
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    history = {'total_loss': [], 'cls_loss': [], 'align_loss': []}
    
    final_epoch = 0
    avg_total = 0 
    for epoch in range(EPOCHS):
        final_epoch = epoch + 1
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        avg_total, avg_cls, avg_align = train_epoch(model, dataloader, optimizer, epoch+1)
        
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
        
        # Step the learning rate scheduler
        scheduler.step(avg_total)
        
        # Save best model
        if avg_total < best_loss and avg_total > 0:
            best_loss = avg_total
            save_model(model, optimizer, scheduler, epoch+1, avg_total, 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, scheduler, epoch+1, avg_total, f'checkpoint_epoch_{epoch+1}.pt')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    save_model(model, optimizer, scheduler, final_epoch, avg_total, 'final_model.pt')
    
    # Save training history
    with open(os.path.join(run_save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved in: {run_save_dir}")
    writer.close()

if __name__ == "__main__":
    main()
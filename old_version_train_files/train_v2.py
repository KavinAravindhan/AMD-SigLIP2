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

TFRECORD_PATH = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/data_xinxin.tfrecord"
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
MAX_TEXT_LEN = 128  # From the debug output, input_ids have length 128

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

IMAGE_HEIGHT = 1055
IMAGE_WIDTH = 703
IMAGE_CHANNELS = 3

description = {
    "input_ids": "int",
    "class": "byte", 
    "normalized_image": "byte"  # Contains 1055×703×3 float32 values as bytes
}

def collate_fn(batch):
    """Custom collate function for TFRecord data"""
    images, labels, input_ids_list = [], [], []
    
    for item in batch:
        try:
            img_bytes = item["normalized_image"]
            img_array = np.frombuffer(img_bytes, dtype=np.float32)
            img = img_array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
            
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
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    if len(images) == 0:
        raise RuntimeError("No valid images in batch! Check your data.")
    
    # Stack everything
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    input_ids = torch.stack(input_ids_list)
    
    # Create attention mask (non-zero tokens)
    attention_mask = (input_ids != 0).bool()
    
    return images, labels, input_ids, attention_mask

dataset = TFRecordDataset(TFRECORD_PATH, None, description)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

class SigLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        self.cls_head = nn.Linear(768, 2)

        self.siglip_loss = SigLIPLoss(
            latent_dim=768,
            text_model="google-t5/t5-base",
            max_txt_len=MAX_TEXT_LEN,  # Now 128
            pool="mean",
            dtype=torch.float32
        )
        
    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_encoder(pixel_values=images).last_hidden_state  # (B, N, 768)
        cls_logits = self.cls_head(img_features[:, 0])  # (B, 2)
        align_loss, _, _ = self.siglip_loss(img_features, input_ids, attention_mask)
        return cls_logits, align_loss

def train_epoch(model, dataloader, optimizer, epoch):
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
        
        # Losses
        cls_loss = F.cross_entropy(cls_logits, labels)
        total_loss_batch = cls_loss + 0.5 * align_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # Accumulate
        total_loss += total_loss_batch.item()
        total_cls += cls_loss.item()
        total_align += align_loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Total={total_loss_batch.item():.4f}, Batch size={len(images)}")
    
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
    }, filepath)
    print(f"Saved: {filepath}")

def main():
    print(f"Starting SigLIP training...")
    print(f"Image dimensions: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Model save directory: {run_save_dir}\n")
    
    # Save training config
    config = {
        'timestamp': timestamp,
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'max_text_len': MAX_TEXT_LEN,
        'image_dimensions': f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}x{IMAGE_CHANNELS}",
        'tfrecord_path': TFRECORD_PATH
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

    for i, item in enumerate(test_dataset):
        if i >= 1:
            break
        img_bytes = item["normalized_image"]
        print(f"First sample: {len(img_bytes)} bytes (expected: {IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS*4} bytes)")
        print(f"Class: {item['class'].decode('utf-8')}")
        print(f"Input IDs length: {len(item['input_ids'])}\n")
    
    print(f"Starting training for {EPOCHS} epochs...")
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
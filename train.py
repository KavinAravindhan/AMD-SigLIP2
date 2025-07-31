import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor, SiglipVisionModel
import numpy as np
import cv2
import random
from datetime import datetime
from tfrecord.torch.dataset import TFRecordDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from embedder import TextEmbedder
from alignment import SigLIPLoss

# Updated TFRecord path
TFRECORD_PATH = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/VQA_v1.tfrecord"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
MAX_TEXT_LEN = 128

# Loss weighting: total_loss = alpha * cls_loss + (1-alpha) * align_loss
# Since text data is currently placeholder, consider increasing ALPHA (e.g., 0.8 or 0.9)
# to focus more on classification until proper text data is available
ALPHA = 0.5  # Default: equal weighting. Increase to 0.8-0.9 if text data is placeholder

# Randomness settings
# Set RANDOM_SEED = 42 for reproducible results (good for debugging/comparison)
# Set RANDOM_SEED = None for true randomness (good for training diversity)
RANDOM_SEED = None  # Change to 42 for reproducibility
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED}")
else:
    print("Using true randomness (no seed set)")

# Original image dimensions from TFRecord
IMAGE_HEIGHT = 703
IMAGE_WIDTH = 1055
IMAGE_CHANNELS = 3

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

# Image augmentation pipeline (light geometric + photometric tweaks, preserve structure)
aug = A.Compose([
    A.HorizontalFlip(p=0.5),  # OK if left/right eye is not labeled
    A.ShiftScaleRotate(
        shift_limit=0.02, 
        scale_limit=0.05, 
        rotate_limit=5,  # <5° to avoid unrealistic tilt
        border_mode=cv2.BORDER_REPLICATE, 
        p=0.6
    ),
    A.RandomBrightnessContrast(0.05, 0.05, p=0.5),
    A.GaussianNoise(var_limit=1e-4, p=0.3),
    A.MotionBlur(blur_limit=3, p=0.15),  # mimics slight acquisition blur
    ToTensorV2()
])

# TFRecord description (verified: 59 images, each with 8 captions)
description = {
    "input_ids": "int",      # Shape: (8, 128) - 8 captions per image, max_len=128
    "attn_mask": "int",      # Shape: (8, 128) - attention masks, 0s indicate padding
    "class": "byte",         # Single byte: 'n' (normal) or 'w' (wet)
    "normalized_image": "byte"  # 8,899,980 bytes = 1055×703×3 float32 values
}

def parse_and_augment_image(img_bytes):
    """Parse normalized image bytes and apply augmentation"""
    try:
        # Parse the float32 array from bytes (original format: 1055×703×3)
        img_array = np.frombuffer(img_bytes, dtype=np.float32)
        img = img_array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).copy()
        
        # Convert from [0,1] float to [0,255] uint8 for albumentations
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Apply augmentation (returns tensor in [0,1] range)
        augmented = aug(image=img_uint8)
        img_tensor = augmented['image'].float()  # Shape: (3, H, W)
        
        return img_tensor
        
    except Exception as e:
        print(f"Error in image processing: {e}")
        # Return a dummy tensor if processing fails
        return torch.zeros((3, 384, 384), dtype=torch.float32)



class UniqueImageDataset:
    """Wrapper to ensure no duplicate images in the same batch"""
    def __init__(self, tfrecord_path, description):
        self.dataset = TFRecordDataset(tfrecord_path, None, description)
        self.items = list(self.dataset)  # Load all items into memory
        self.used_indices = set()
        
    def __len__(self):
        return len(self.items)
    
    def get_batch_items(self, batch_size):
        """Get a batch of unique items"""
        if len(self.used_indices) >= len(self.items):
            # Reset if we've used all items
            self.used_indices.clear()
        
        batch_items = []
        available_indices = [i for i in range(len(self.items)) if i not in self.used_indices]
        
        # Randomly sample without replacement
        selected_indices = random.sample(available_indices, min(batch_size, len(available_indices)))
        
        for idx in selected_indices:
            batch_items.append(self.items[idx])
            self.used_indices.add(idx)
        
        return batch_items

def collate_fn(batch):
    """Custom collate function with multiple caption selection"""
    images, labels, input_ids_list, attention_masks = [], [], [], []
    
    for item in batch:
        try:
            # Process image with augmentation
            img_bytes = item["normalized_image"]
            img_tensor = parse_and_augment_image(img_bytes)
            
            # Resize to SigLIP expected size (384x384)
            if img_tensor.shape[1] != 384 or img_tensor.shape[2] != 384:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0), 
                    size=(384, 384), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Normalize for SigLIP: [0,1] -> [-1,1]
            img_tensor = (img_tensor - 0.5) / 0.5
            images.append(img_tensor)
            
            # Process label
            class_str = item["class"].decode('utf-8')
            label = 0 if class_str == 'n' else 1  # n=normal=0, w=wet=1
            labels.append(label)
            
            # Handle multiple captions: input_ids shape is (8, 128)
            all_input_ids = np.array(item["input_ids"])  # Shape: (8, 128)
            all_attn_masks = np.array(item["attn_mask"])  # Shape: (8, 128)
            
            # Verify expected shape
            if all_input_ids.shape != (8, 128) or all_attn_masks.shape != (8, 128):
                print(f"⚠️  Unexpected shape: input_ids={all_input_ids.shape}, attn_mask={all_attn_masks.shape}")
            
            # Randomly select one of the 8 captions
            caption_idx = random.randint(0, 7)
            selected_input_ids = all_input_ids[caption_idx]  # Shape: (128,)
            selected_attn_mask = all_attn_masks[caption_idx]  # Shape: (128,)
            
            # Convert to tensors
            input_ids = torch.tensor(selected_input_ids, dtype=torch.long)
            attention_mask = torch.tensor(selected_attn_mask, dtype=torch.bool)
            
            input_ids_list.append(input_ids)
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

def create_dataloader():
    """Create DataLoader with unique image constraint for SigLIP"""
    unique_dataset = UniqueImageDataset(TFRECORD_PATH, description)
    
    # Create custom DataLoader that ensures unique images per batch
    class CustomDataLoader:
        def __init__(self, dataset, batch_size):
            self.dataset = dataset
            self.batch_size = batch_size
        
        def __iter__(self):
            while True:
                batch_items = self.dataset.get_batch_items(self.batch_size)
                if len(batch_items) == 0:
                    break
                
                # Process batch through collate_fn
                try:
                    yield collate_fn(batch_items)
                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    continue
        
        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    return CustomDataLoader(unique_dataset, BATCH_SIZE)

class SigLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ USING GEMMA3's SigLIP encoder: google/siglip-so400m-patch14-384
        # This model outputs 1152-dimensional features (NOT 768 like older models)
        self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        
        # ✅ Classification head uses 1152 input dimensions (matches Gemma3 SigLIP)
        self.cls_head = nn.Linear(1152, 2)  # 1152 -> 2 classes (normal/wet)

        # ✅ SigLIP loss configured for 1152 latent dimensions (Gemma3 compatible)
        self.siglip_loss = SigLIPLoss(
            latent_dim=1152,  # ← GEMMA3 SigLIP dimension (NOT 768)
            text_model="google-t5/t5-base",
            max_txt_len=MAX_TEXT_LEN,
            pool="mean",
            dtype=torch.float32
        )
        
    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass with tokenized inputs
        Args:
            images: (B, 3, 384, 384) tensor
            input_ids: (B, seq_len) tensor of token ids
            attention_mask: (B, seq_len) tensor of attention mask
        """
        # ✅ Extract 1152-dimensional features from Gemma3's SigLIP encoder
        img_features = self.image_encoder(pixel_values=images).last_hidden_state  # (B, 729, 1152)
        
        # Classification: Use CLS token (first token) with 1152 dimensions
        cls_logits = self.cls_head(img_features[:, 0])  # (B, 1152) -> (B, 2)
        
        # SigLIP alignment loss with 1152-dimensional features
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
    print(f"✅ USING GEMMA3's SigLIP ENCODER:")
    print(f"   Model: google/siglip-so400m-patch14-384")
    print(f"   Latent dimension: 1152 (NOT 768)")
    print(f"   Input image size: 384x384")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Alpha (loss weighting): {ALPHA}")
    print(f"TFRecord path: {TFRECORD_PATH}")
    print(f"Image augmentation: Enabled (geometric + photometric)")
    print(f"Random seed: {RANDOM_SEED if RANDOM_SEED is not None else 'None (true randomness)'}")
    print(f"SigLIP constraint: Unique images per batch")
    print(f"Note: Multiple caption support pending - currently using original format")
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
        'tfrecord_path': TFRECORD_PATH,
        'siglip_model': 'google/siglip-so400m-patch14-384',
        'input_image_size': '384x384',
        'original_image_size': f'{IMAGE_WIDTH}x{IMAGE_HEIGHT}x{IMAGE_CHANNELS}',
        'image_normalization': 'mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)',
        'latent_dim': 1152,
        'augmentation': 'albumentations (geometric + photometric)',
        'random_seed': RANDOM_SEED,
        'unique_images_per_batch': True,
        'multiple_captions': 'pending_implementation'
    }
    
    with open(os.path.join(run_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model and dataloader
    model = SigLIPModel().to(DEVICE)
    
    # ✅ VERIFY GEMMA3 SigLIP DIMENSIONS
    print(f"\n🔍 Model Architecture Verification:")
    print(f"   Image encoder: {model.image_encoder.__class__.__name__}")
    print(f"   Image encoder config: {model.image_encoder.config.hidden_size}D features")
    print(f"   Classification head: {model.cls_head.in_features} -> {model.cls_head.out_features}")
    print(f"   SigLIP loss latent_dim: {model.siglip_loss.latent_dim}")
    print(f"   ✅ All components use 1152 dimensions (Gemma3 compatible)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    dataloader = create_dataloader()
    
    print("Checking dataset...")
    # Quick dataset inspection
    try:
        sample_dataset = UniqueImageDataset(TFRECORD_PATH, description)
        print(f"Total samples in dataset: {len(sample_dataset)}")
        print(f"Expected number of batches per epoch: {len(dataloader)}")
        
        # Inspect first sample
        if len(sample_dataset.items) > 0:
            first_item = sample_dataset.items[0]
            print(f"First sample keys: {list(first_item.keys())}")
            print(f"Normalized image bytes length: {len(first_item.get('normalized_image', b''))}")
            print(f"Expected bytes: {IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * 4}")  # 4 bytes per float32
            print(f"Class: {first_item.get('class', b'').decode('utf-8')}")
            print(f"Input IDs length: {len(first_item.get('input_ids', []))}")
            print(f"Attention mask length: {len(first_item.get('attn_mask', []))}")
            
    except Exception as e:
        print(f"Dataset inspection error: {e}")
    
    print(f"\nStarting training for {EPOCHS} epochs...")
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
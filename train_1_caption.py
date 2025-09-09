import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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

# nohup python train_1_caption.py > /home/kavin/AMD-SigLIP2/terminal_output_nohup/train_1_caption.txt 2>&1 &

# Updated TFRecord path for v4
TFRECORD_PATH = "/media/16TB_Storage/kavin/amd_siglip/data_tfrecords/VQA_v4.tfrecord"

# Best hyperparameters from optimization
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1.6079555710533247e-06
ALPHA = 0.8840963962895334
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.7851246675328261
EARLY_STOPPING_PATIENCE = 20
DROPOUT_RATE = 0.057129660535791646
MAX_TEXT_LEN = 128

# Random sampling settings
RANDOM_SAMPLES_PER_EPOCH = 1000
RANDOM_SEED = None
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

# Original image dimensions
IMAGE_HEIGHT = 703
IMAGE_WIDTH = 1055
IMAGE_CHANNELS = 3

# Model save directory
MODEL_SAVE_DIR = "/media/16TB_Storage/kavin/amd_siglip/saved_models/VQA_v4_1_caption"
RUNS_SAVE_DIR = "/media/16TB_Storage/kavin/amd_siglip/runs"

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
run_save_dir = os.path.join(RUNS_SAVE_DIR, f"run_{timestamp}")
os.makedirs(run_save_dir, exist_ok=True)

# TensorBoard
writer = SummaryWriter(f'runs/siglip_training_{timestamp}')

# Optimized image augmentation pipeline
aug = A.Compose([
    A.HorizontalFlip(p=0.9674733435973407),
    A.ShiftScaleRotate(
        shift_limit=0.028795483291628815,
        scale_limit=0.037814873748683406,
        rotate_limit=3,
        border_mode=cv2.BORDER_REPLICATE,
        p=0.028486520024779284
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.09474000592838613,
        contrast_limit=0.03187889420730783,
        p=0.060697129192364106
    ),
    A.GaussNoise(noise_limit=(0, 1e-4), p=0.342779329951288),
    A.MotionBlur(blur_limit=3, p=0.194405815151957),
    ToTensorV2()
])

# TFRecord description for v4 with 1 caption
description = {
    "input_ids": "int",
    "input_ids_shape": "int",
    "attn_mask": "int",
    "attn_mask_shape": "int",
    "class": "byte",
    "normalized_image": "byte"
}

# Caption settings for v4 (1 caption per image)
NUM_CAPTIONS_PER_IMAGE = 1
CAPTION_MAX_LENGTH = 128

def parse_and_augment_image(img_bytes):
    """Parse normalized image bytes and apply augmentation"""
    try:
        img_array = np.frombuffer(img_bytes, dtype=np.float32)
        img = img_array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).copy()
        
        img_uint8 = (img * 255).astype(np.uint8)
        augmented = aug(image=img_uint8)
        img_tensor = augmented['image'].float()
        
        return img_tensor
        
    except Exception as e:
        print(f"Error in image processing: {e}")
        return torch.zeros((3, 384, 384), dtype=torch.float32)

class RandomSampleDataset(Dataset):
    """Dataset that randomly samples from TFRecord with replacement"""
    def __init__(self, tfrecord_path, description, samples_per_epoch):
        self.dataset = TFRecordDataset(tfrecord_path, None, description)
        self.items = list(self.dataset)
        self.samples_per_epoch = samples_per_epoch
        print(f"Loaded {len(self.items)} items from TFRecord")
        print(f"Will generate {samples_per_epoch} random samples per epoch")
        
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        # Randomly sample from available items with replacement
        random_idx = random.randint(0, len(self.items) - 1)
        return self.items[random_idx]

def collate_fn(batch):
    """Custom collate function for TFRecord v4 format with 1 caption"""
    images, labels, input_ids_list, attention_masks = [], [], [], []
    
    for item in batch:
        try:
            # Process image with augmentation
            img_bytes = item["normalized_image"]
            img_tensor = parse_and_augment_image(img_bytes)
            
            # Resize to SigLIP expected size
            if img_tensor.shape[1] != 384 or img_tensor.shape[2] != 384:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0), 
                    size=(384, 384), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Normalize for SigLIP
            img_tensor = (img_tensor - 0.5) / 0.5
            images.append(img_tensor)
            
            # Process label
            class_str = item["class"].decode('utf-8')
            label = 0 if class_str == 'n' else 1
            labels.append(label)
            
            # Reconstruct arrays from flattened data
            input_ids_flat = np.array(item["input_ids"])
            input_ids_shape = tuple(item["input_ids_shape"])
            attn_mask_flat = np.array(item["attn_mask"]) 
            attn_mask_shape = tuple(item["attn_mask_shape"])
            
            input_ids_array = input_ids_flat.reshape(input_ids_shape)
            attn_mask_array = attn_mask_flat.reshape(attn_mask_shape)
            
            # For v4 with 1 caption, just take the single caption
            if input_ids_array.shape[0] == 1:
                selected_input_ids = input_ids_array[0]
                selected_attn_mask = attn_mask_array[0]
            else:
                # Fallback for unexpected format
                selected_input_ids = input_ids_array.flatten()[:CAPTION_MAX_LENGTH]
                selected_attn_mask = attn_mask_array.flatten()[:CAPTION_MAX_LENGTH]
                if len(selected_input_ids) < CAPTION_MAX_LENGTH:
                    pad_len = CAPTION_MAX_LENGTH - len(selected_input_ids)
                    selected_input_ids = np.pad(selected_input_ids, (0, pad_len), 'constant')
                    selected_attn_mask = np.pad(selected_attn_mask, (0, pad_len), 'constant')
            
            input_ids = torch.tensor(selected_input_ids, dtype=torch.long)
            attention_mask = torch.tensor(selected_attn_mask, dtype=torch.bool)
            
            input_ids_list.append(input_ids)
            attention_masks.append(attention_mask)
            
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    if len(images) == 0:
        raise RuntimeError("No valid images in batch!")
    
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_masks)
    
    return images, labels, input_ids, attention_mask

class SigLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        
        # Add dropout
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.cls_head = nn.Linear(1152, 2)

        self.siglip_loss = SigLIPLoss(
            latent_dim=1152,
            text_model="google-t5/t5-base",
            max_txt_len=MAX_TEXT_LEN,
            pool="mean",
            dtype=torch.float32
        )
        
    def forward(self, images, input_ids, attention_mask):
        img_features = self.image_encoder(pixel_values=images).last_hidden_state
        
        # Apply dropout before classification
        cls_features = self.dropout(img_features[:, 0])
        cls_logits = self.cls_head(cls_features)
        
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
        
        cls_logits, align_loss = model(images, input_ids, attention_mask)
        
        cls_loss = F.cross_entropy(cls_logits, labels)
        total_loss_batch = ALPHA * cls_loss + (1 - ALPHA) * align_loss
        
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_cls += cls_loss.item()
        total_align += align_loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Total={total_loss_batch.item():.4f}, Cls={cls_loss.item():.4f}, Align={align_loss.item():.4f}")
    
    if num_batches == 0:
        return 0, 0, 0
    
    return total_loss/num_batches, total_cls/num_batches, total_align/num_batches

def save_model(model, optimizer, scheduler, epoch, loss, filename):
    filepath = os.path.join(MODEL_SAVE_DIR, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'alpha': ALPHA,
    }, filepath)
    print(f"Saved: {filepath}")

def main():
    print(f"Training Configuration:")
    print(f"TFRecord: {TFRECORD_PATH}")
    print(f"Random samples per epoch: {RANDOM_SAMPLES_PER_EPOCH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Alpha: {ALPHA}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Device: {DEVICE}")
    
    # Save config
    config = {
        'timestamp': timestamp,
        'tfrecord_path': TFRECORD_PATH,
        'random_samples_per_epoch': RANDOM_SAMPLES_PER_EPOCH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'alpha': ALPHA,
        'dropout_rate': DROPOUT_RATE,
        'scheduler_patience': SCHEDULER_PATIENCE,
        'scheduler_factor': SCHEDULER_FACTOR,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'captions_per_image': NUM_CAPTIONS_PER_IMAGE,
        'image_encoder': 'google/siglip-so400m-patch14-384',
        'text_model': 'google-t5/t5-base'
    }
    
    with open(os.path.join(MODEL_SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataset and dataloader
    dataset = RandomSampleDataset(TFRECORD_PATH, description, RANDOM_SAMPLES_PER_EPOCH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = SigLIPModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    best_loss = float('inf')
    patience_counter = 0
    history = {'total_loss': [], 'cls_loss': [], 'align_loss': []}
    
    for epoch in range(EPOCHS):
        avg_total, avg_cls, avg_align = train_epoch(model, dataloader, optimizer, epoch+1)
        
        history['total_loss'].append(avg_total)
        history['cls_loss'].append(avg_cls)
        history['align_loss'].append(avg_align)
        
        print(f"Epoch {epoch+1}: Total Loss: {avg_total:.4f}, Cls Loss: {avg_cls:.4f}, Align Loss: {avg_align:.4f}")
        
        writer.add_scalar('Loss/Total', avg_total, epoch)
        writer.add_scalar('Loss/Classification', avg_cls, epoch)
        writer.add_scalar('Loss/Alignment', avg_align, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step(avg_total)
        
        if avg_total < best_loss and avg_total > 0:
            best_loss = avg_total
            save_model(model, optimizer, scheduler, epoch+1, avg_total, 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, scheduler, epoch+1, avg_total, f'checkpoint_epoch_{epoch+1}.pt')
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    save_model(model, optimizer, scheduler, epoch+1, avg_total, 'final_model.pt')
    
    with open(os.path.join(MODEL_SAVE_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training completed! Best loss: {best_loss:.4f}")
    print(f"Models saved in: {MODEL_SAVE_DIR}")
    writer.close()

if __name__ == "__main__":
    main()
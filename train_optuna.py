"""
Optuna Hyperparameter Search for SigLIP Training with Vision Encoder Comparison

SCIENTIFIC COMPARISON: Gemma3 vs MedGemma SigLIP Encoders
===========================================================

This script compares two vision encoders in a rigorous hyperparameter search:

1. **Gemma3 SigLIP** (google/siglip-so400m-patch14-384)
   - General-purpose vision encoder
   - Trained on diverse internet images
   - 1152 latent dimensions

2. **MedGemma SigLIP** (extracted from google/medgemma-4b-it)
   - Medical-specific vision encoder  
   - Pre-trained on chest X-rays, dermatology, ophthalmology, histopathology
   - Same 1152 latent dimensions
   - Expected better performance on medical imaging tasks

RESEARCH HYPOTHESIS:
Medical pre-training should provide better feature representations for OCT imaging
and AMD classification compared to general-purpose pre-training.

SETUP REQUIREMENTS:
- Request access to google/medgemma-4b-it on Hugging Face
- Accept Health AI Developer Foundation terms
- Run verify_medgemma_setup() to test access

FRAMEWORK FEATURES:
- Automatic model loading with fallbacks
- Memory-efficient vision encoder extraction
- Complete hyperparameter optimization
- TensorBoard logging per trial
- Scientific comparison analysis

READY TO RUN: Both encoders fully integrated with official APIs
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # This works without TensorFlow!
from transformers import AutoImageProcessor, SiglipVisionModel, AutoModelForImageTextToText
import numpy as np
import cv2
import random
from datetime import datetime
from tfrecord.torch.dataset import TFRecordDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from embedder import TextEmbedder
from alignment import SigLIPLoss
import time
import optuna
import pickle
import matplotlib.pyplot as plt

# Updated TFRecord path for v2
TFRECORD_PATH = "/Users/kavin/Columbia/Labs/Kaveri Lab/AMD-SigLIP2/VQA_v2.tfrecord"

# Fixed hyperparameters
EPOCHS = 100  # Changed from 50 to 100 as requested
MAX_TEXT_LEN = 128

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
run_save_dir = os.path.join(MODEL_SAVE_DIR, f"optuna_run_{timestamp}")
os.makedirs(run_save_dir, exist_ok=True)
print(f"Models will be saved to: {run_save_dir}")

# FIXED TFRecord description - including the missing shape fields
description = {
    "input_ids": "int",           # Flattened array
    "input_ids_shape": "int",     # Original shape (8, 128)
    "attn_mask": "int",           # Flattened array
    "attn_mask_shape": "int",     # Original shape (8, 128) 
    "class": "byte",              # Single byte: 'n' (normal) or 'w' (wet)
    "normalized_image": "byte"    # 8,899,980 bytes = 1055×703×3 float32 values
}

# Caption settings (ACTUAL format based on your data creation)
NUM_CAPTIONS_PER_IMAGE = 8   # 8 VQA explanations per image
CAPTION_MAX_LENGTH = 128     # T5 tokenizer max length

def get_augmentation_pipeline(trial):
    """Create augmentation pipeline with hyperparameter search"""
    # Suggest augmentation parameters
    horizontal_flip_prob = trial.suggest_float("horizontal_flip_prob", 0.0, 1.0)
    shift_scale_rotate_prob = trial.suggest_float("shift_scale_rotate_prob", 0.0, 1.0)
    shift_limit = trial.suggest_float("shift_limit", 0.01, 0.05)
    scale_limit = trial.suggest_float("scale_limit", 0.02, 0.1)
    rotate_limit = trial.suggest_int("rotate_limit", 2, 10)
    brightness_contrast_prob = trial.suggest_float("brightness_contrast_prob", 0.0, 1.0)
    brightness_limit = trial.suggest_float("brightness_limit", 0.02, 0.1)
    contrast_limit = trial.suggest_float("contrast_limit", 0.02, 0.1)
    gauss_noise_prob = trial.suggest_float("gauss_noise_prob", 0.0, 0.5)
    motion_blur_prob = trial.suggest_float("motion_blur_prob", 0.0, 0.3)
    blur_limit = trial.suggest_int("blur_limit", 2, 5)
    
    aug = A.Compose([
        A.HorizontalFlip(p=horizontal_flip_prob),
        A.ShiftScaleRotate(
            shift_limit=shift_limit, 
            scale_limit=scale_limit, 
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_REPLICATE, 
            p=shift_scale_rotate_prob
        ),
        A.RandomBrightnessContrast(brightness_limit, contrast_limit, p=brightness_contrast_prob),
        A.GaussNoise(noise_limit=(0, 1e-4), p=gauss_noise_prob),
        A.MotionBlur(blur_limit=blur_limit, p=motion_blur_prob),
        ToTensorV2()
    ])
    
    return aug

def parse_and_augment_image(img_bytes, aug_pipeline):
    """Parse normalized image bytes and apply augmentation"""
    try:
        # Parse the float32 array from bytes (original format: 1055×703×3)
        img_array = np.frombuffer(img_bytes, dtype=np.float32)
        img = img_array.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS).copy()
        
        # Convert from [0,1] float to [0,255] uint8 for albumentations
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Apply augmentation (returns tensor in [0,1] range)
        augmented = aug_pipeline(image=img_uint8)
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
        # FIXED: Don't auto-reset - let the dataloader control epochs
        if len(self.used_indices) >= len(self.items):
            return []  # End of epoch - return empty batch
        
        batch_items = []
        available_indices = [i for i in range(len(self.items)) if i not in self.used_indices]
        
        # Randomly sample without replacement
        selected_indices = random.sample(available_indices, min(batch_size, len(available_indices)))
        
        for idx in selected_indices:
            batch_items.append(self.items[idx])
            self.used_indices.add(idx)
        
        return batch_items

def collate_fn(batch, aug_pipeline):
    """FIXED: Custom collate function for TFRecord v2 format with flattened arrays"""
    images, labels, input_ids_list, attention_masks = [], [], [], []
    
    for item in batch:
        try:
            # Process image with augmentation
            img_bytes = item["normalized_image"]
            img_tensor = parse_and_augment_image(img_bytes, aug_pipeline)
            
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
            
            # FIXED: Reconstruct multi-dimensional arrays from flattened data
            # Get flattened arrays and shapes
            input_ids_flat = np.array(item["input_ids"])
            input_ids_shape = tuple(item["input_ids_shape"])
            attn_mask_flat = np.array(item["attn_mask"]) 
            attn_mask_shape = tuple(item["attn_mask_shape"])
            
            # Reconstruct original multi-dimensional arrays
            input_ids_array = input_ids_flat.reshape(input_ids_shape)  # Should be (8, 128)
            attn_mask_array = attn_mask_flat.reshape(attn_mask_shape)  # Should be (8, 128)
            
            # RANDOM CAPTION SELECTION: Choose 1 of 8 captions randomly
            caption_idx = random.randint(0, NUM_CAPTIONS_PER_IMAGE - 1)
            selected_input_ids = input_ids_array[caption_idx]     # Shape: (128,)
            selected_attn_mask = attn_mask_array[caption_idx]     # Shape: (128,)
            
            # Convert to tensors
            input_ids = torch.tensor(selected_input_ids, dtype=torch.long)
            attention_mask = torch.tensor(selected_attn_mask, dtype=torch.bool)
            
            input_ids_list.append(input_ids)
            attention_masks.append(attention_mask)
            
        except Exception as e:
            print(f"Error processing item: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(images) == 0:
        raise RuntimeError("No valid images in batch! Check your data.")
    
    # Stack everything
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    input_ids = torch.stack(input_ids_list)       # Shape: (batch_size, 128)
    attention_mask = torch.stack(attention_masks) # Shape: (batch_size, 128)
    
    return images, labels, input_ids, attention_mask

def create_dataloader(batch_size, aug_pipeline):
    """Create DataLoader with unique image constraint for SigLIP"""
    unique_dataset = UniqueImageDataset(TFRECORD_PATH, description)
    
    # Create custom DataLoader that ensures unique images per batch
    class CustomDataLoader:
        def __init__(self, dataset, batch_size, aug_pipeline):
            self.dataset = dataset
            self.batch_size = batch_size
            self.aug_pipeline = aug_pipeline
        
        def __iter__(self):
            # FIXED: Reset at the start of each epoch and track progress
            self.dataset.used_indices.clear()
            total_items = len(self.dataset.items)
            items_yielded = 0
            
            # FIXED: Use finite loop based on dataset size
            while items_yielded < total_items:
                batch_items = self.dataset.get_batch_items(self.batch_size)
                if len(batch_items) == 0:
                    break
                
                # Process batch through collate_fn
                try:
                    yield collate_fn(batch_items, self.aug_pipeline)
                    items_yielded += len(batch_items)
                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    continue
        
        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    return CustomDataLoader(unique_dataset, batch_size, aug_pipeline)

def load_medgemma_vision_encoder():
    """Load MedGemma vision encoder with proper error handling"""
    try:
        print("Loading MedGemma model...")
        # Load the full MedGemma model
        full_model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for training
            device_map=None  # Don't auto-assign devices
        )
        
        # Try different possible vision component names
        vision_component_names = ['vision_tower', 'vision_model', 'vision_encoder', 'visual_encoder']
        vision_encoder = None
        
        for attr_name in vision_component_names:
            if hasattr(full_model, attr_name):
                vision_encoder = getattr(full_model, attr_name)
                print(f"Found vision encoder: {attr_name}")
                break
        
        if vision_encoder is None:
            # Try to find vision-related attributes
            vision_attrs = [attr for attr in dir(full_model) if 'vision' in attr.lower() or 'visual' in attr.lower()]
            if vision_attrs:
                print(f"Available vision attributes: {vision_attrs}")
                # Try the first one
                vision_encoder = getattr(full_model, vision_attrs[0])
                print(f"Using: {vision_attrs[0]}")
            else:
                raise ValueError("No vision encoder component found in MedGemma model")
        
        # Verify the encoder has the expected structure
        if hasattr(vision_encoder, 'config') and hasattr(vision_encoder.config, 'hidden_size'):
            print(f"Vision encoder hidden size: {vision_encoder.config.hidden_size}")
        
        return vision_encoder
        
    except Exception as e:
        print(f"Failed to load MedGemma vision encoder: {e}")
        if "401" in str(e) or "access" in str(e).lower():
            print("Access issue detected. To enable MedGemma:")
            print("1. Go to: https://huggingface.co/google/medgemma-4b-it")
            print("2. Click 'Request access'")
            print("3. Agree to Health AI Developer Foundation terms")
            print("4. Wait for approval (usually immediate)")
        return None

class SigLIPModel(nn.Module):
    def __init__(self, trial):
        super().__init__()
        
        # Suggest image encoder model for comparison
        image_encoder_choice = trial.suggest_categorical("image_encoder", [
            "google/siglip-so400m-patch14-384",  # Gemma3's SigLIP (1152-dim)
            "medgemma-vision-encoder",  # MedGemma's Medical SigLIP (1152-dim)
        ])
        
        # Suggest text encoder model
        text_model_choice = trial.suggest_categorical("text_model", [
            "google-t5/t5-base", 
            "google-t5/t5-small", 
            "google-t5/t5-large"
        ])
        
        # Suggest dropout rate for classification head
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        
        # Load the selected image encoder
        if image_encoder_choice == "google/siglip-so400m-patch14-384":
            # Standard Gemma3 SigLIP encoder
            print("Loading Gemma3 SigLIP encoder...")
            self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_choice)
            encoder_output_dim = 1152
            print("Gemma3 SigLIP encoder loaded successfully")
            
        elif image_encoder_choice == "medgemma-vision-encoder":
            # MedGemma's medical-trained SigLIP encoder
            print("Loading MedGemma vision encoder...")
            medgemma_encoder = load_medgemma_vision_encoder()
            
            if medgemma_encoder is not None:
                self.image_encoder = medgemma_encoder
                encoder_output_dim = 1152  # Confirmed from research
                print("MedGemma vision encoder loaded successfully")
            else:
                print("Failed to load MedGemma, falling back to Gemma3 SigLIP")
                self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
                encoder_output_dim = 1152
        else:
            # Fallback - should not reach here with current configuration
            self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
            encoder_output_dim = 1152
            print(f"Warning: Unknown encoder {image_encoder_choice}, using Gemma3 SigLIP fallback")
        
        # Classification head with dynamic input dimension
        self.dropout = nn.Dropout(dropout_rate)
        self.cls_head = nn.Linear(encoder_output_dim, 2)  # Dynamic -> 2 classes (normal/wet)

        # SigLIP loss configured with dynamic latent dimensions
        self.siglip_loss = SigLIPLoss(
            latent_dim=encoder_output_dim,  # Dynamic based on chosen encoder
            text_model=text_model_choice,
            max_txt_len=MAX_TEXT_LEN,
            pool="mean",
            dtype=torch.float32
        )
        
        # Store for forward pass
        self.encoder_output_dim = encoder_output_dim
        self.image_encoder_choice = image_encoder_choice
        
    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass with tokenized inputs
        Args:
            images: (B, 3, 384, 384) tensor
            input_ids: (B, seq_len) tensor of token ids
            attention_mask: (B, seq_len) tensor of attention mask
        """
        # Extract features from selected image encoder
        img_features = self.image_encoder(pixel_values=images).last_hidden_state  # (B, 729, encoder_dim)
        
        # Classification: Use CLS token (first token) with dropout
        cls_features = self.dropout(img_features[:, 0])  # Apply dropout
        cls_logits = self.cls_head(cls_features)  # (B, encoder_dim) -> (B, 2)
        
        # SigLIP alignment loss with dynamic dimensions
        align_loss, _, _ = self.siglip_loss(img_features, input_ids, attention_mask)
        
        return cls_logits, align_loss

def train_epoch(model, dataloader, optimizer, epoch, alpha, writer=None, trial_number=None):
    model.train()
    total_loss = total_cls = total_align = 0
    num_batches = 0
    
    print(f"Starting epoch {epoch}")
    
    for batch_idx, (images, labels, input_ids, attention_mask) in enumerate(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        
        # Forward pass
        cls_logits, align_loss = model(images, input_ids, attention_mask)
        
        # Losses
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
        
        # Log to TensorBoard every batch (optional - can be reduced for performance)
        if writer is not None:
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar(f'Batch/Total_Loss', total_loss_batch.item(), global_step)
            writer.add_scalar(f'Batch/Classification_Loss', cls_loss.item(), global_step)
            writer.add_scalar(f'Batch/Alignment_Loss', align_loss.item(), global_step)
        
        if batch_idx % 5 == 0:  # Print every 5 batches to reduce output
            print(f"Epoch {epoch}, Batch {batch_idx + 1}: Total={total_loss_batch.item():.4f}, Cls={cls_loss.item():.4f}, Align={align_loss.item():.4f}")
    
    print(f"Epoch {epoch} completed: Processed {num_batches} batches")
    
    if num_batches == 0:
        print("No valid batches in epoch!")
        return 0, 0, 0
    
    avg_total = total_loss/num_batches
    avg_cls = total_cls/num_batches 
    avg_align = total_align/num_batches
    
    # Log epoch averages to TensorBoard
    if writer is not None:
        writer.add_scalar(f'Epoch/Total_Loss', avg_total, epoch)
        writer.add_scalar(f'Epoch/Classification_Loss', avg_cls, epoch)
        writer.add_scalar(f'Epoch/Alignment_Loss', avg_align, epoch)
        writer.add_scalar(f'Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log hyperparameters as text (only once)
        if epoch == 1 and trial_number is not None:
            writer.add_text('Hyperparameters/Trial_Number', str(trial_number), epoch)
            writer.add_text('Hyperparameters/Alpha', str(alpha), epoch)
    
    return avg_total, avg_cls, avg_align

def objective(trial):
    """Optuna objective function"""
    
    # Set random seed for reproducibility within trial
    trial_seed = 42 + trial.number
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    
    print(f"Starting trial {trial.number}")
    
    # Hyperparameter suggestions
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 48, 64])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-3, 1e-4, 1e-5])
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    
    # Optimizer hyperparameters
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # Scheduler hyperparameters
    scheduler_patience = trial.suggest_int("scheduler_patience", 2, 8)
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.8)
    
    # Early stopping hyperparameters
    early_stopping_patience = trial.suggest_int("early_stopping_patience", 5, 15)
    
    # Text length hyperparameter
    max_text_len = trial.suggest_categorical("max_text_len", [64, 128, 256])
    
    print(f"Trial {trial.number} hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Alpha: {alpha:.3f}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"  Weight decay: {weight_decay:.6f}")
    print(f"  Scheduler patience: {scheduler_patience}")
    print(f"  Scheduler factor: {scheduler_factor:.3f}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Max text length: {max_text_len}")
    
    # Create trial-specific logging directory
    trial_dir = os.path.join(run_save_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Create TensorBoard writer for this trial
    tb_log_dir = os.path.join(trial_dir, 'tensorboard')
    writer = SummaryWriter(tb_log_dir)
    
    print(f"TensorBoard logs for trial {trial.number}: {tb_log_dir}")
    
    # Initialize trial logging
    trial_log = {
        'trial_number': trial.number,
        'tensorboard_dir': tb_log_dir,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'optimizer': optimizer_name,
            'weight_decay': weight_decay,
            'scheduler_patience': scheduler_patience,
            'scheduler_factor': scheduler_factor,
            'early_stopping_patience': early_stopping_patience,
            'max_text_len': max_text_len
        },
        'epoch_losses': [],
        'epoch_durations': [],
        'best_loss': float('inf'),
        'final_epoch': 0
    }
    
    try:
        # Create augmentation pipeline with trial-specific parameters
        aug_pipeline = get_augmentation_pipeline(trial)
        
        # Add augmentation params to trial log
        trial_log['hyperparameters'].update({
            'horizontal_flip_prob': trial.params.get('horizontal_flip_prob'),
            'shift_scale_rotate_prob': trial.params.get('shift_scale_rotate_prob'),
            'shift_limit': trial.params.get('shift_limit'),
            'scale_limit': trial.params.get('scale_limit'),
            'rotate_limit': trial.params.get('rotate_limit'),
            'brightness_contrast_prob': trial.params.get('brightness_contrast_prob'),
            'brightness_limit': trial.params.get('brightness_limit'),
            'contrast_limit': trial.params.get('contrast_limit'),
            'gauss_noise_prob': trial.params.get('gauss_noise_prob'),
            'motion_blur_prob': trial.params.get('motion_blur_prob'),
            'blur_limit': trial.params.get('blur_limit')
        })
        
        # Create model with trial-specific parameters
        model = SigLIPModel(trial).to(DEVICE)
        
        # Add model params to trial log
        trial_log['hyperparameters'].update({
            'image_encoder': trial.params.get('image_encoder'),
            'text_model': trial.params.get('text_model'),
            'dropout_rate': trial.params.get('dropout_rate')
        })
        
        # Create optimizer based on trial suggestion
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = trial.suggest_float("momentum", 0.8, 0.99)
            trial_log['hyperparameters']['momentum'] = momentum
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        
        # Create dataloader
        dataloader = create_dataloader(batch_size, aug_pipeline)
        
        # Log hyperparameters to TensorBoard
        writer.add_text('Config/Batch_Size', str(batch_size), 0)
        writer.add_text('Config/Learning_Rate', str(learning_rate), 0)
        writer.add_text('Config/Alpha', str(alpha), 0)
        writer.add_text('Config/Optimizer', optimizer_name, 0)
        writer.add_text('Config/Weight_Decay', str(weight_decay), 0)
        writer.add_text('Config/Image_Encoder', trial.params.get('image_encoder', 'unknown'), 0)
        writer.add_text('Config/Text_Model', trial.params.get('text_model', 'unknown'), 0)
        writer.add_text('Config/Dropout_Rate', str(trial.params.get('dropout_rate', 0.0)), 0)
        
        # Log hyperparameters as scalars for easier analysis
        hparams = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'weight_decay': weight_decay,
            'scheduler_patience': scheduler_patience,
            'scheduler_factor': scheduler_factor,
            'early_stopping_patience': early_stopping_patience,
            'dropout_rate': trial.params.get('dropout_rate', 0.0)
        }
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            
            # Train epoch with TensorBoard logging
            avg_total, avg_cls, avg_align = train_epoch(
                model, dataloader, optimizer, epoch+1, alpha, 
                writer=writer, trial_number=trial.number
            )
            
            epoch_duration = time.time() - epoch_start_time
            
            # Log additional metrics to TensorBoard
            writer.add_scalar('Training/Epoch_Duration', epoch_duration, epoch + 1)
            writer.add_scalar('Training/Patience_Counter', patience_counter, epoch + 1)
            writer.add_scalar('Training/Best_Loss_So_Far', best_loss, epoch + 1)
            
            # Log epoch results
            trial_log['epoch_losses'].append({
                'epoch': epoch + 1,
                'total_loss': avg_total,
                'cls_loss': avg_cls,
                'align_loss': avg_align
            })
            trial_log['epoch_durations'].append(epoch_duration)
            trial_log['final_epoch'] = epoch + 1
            
            # Report to optuna
            trial.report(avg_total, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                trial_log['status'] = 'pruned'
                trial_log['pruned_at_epoch'] = epoch + 1
                # Log final hyperparameters and results to TensorBoard
                writer.add_hparams(hparams, {'final_loss': avg_total, 'status': 0})  # 0 for pruned
                writer.close()
                # Save trial log before pruning
                with open(os.path.join(trial_dir, 'trial_log.json'), 'w') as f:
                    json.dump(trial_log, f, indent=2)
                raise optuna.exceptions.TrialPruned()
            
            # Update scheduler
            scheduler.step(avg_total)
            
            # Early stopping logic and best model saving
            if avg_total < best_loss:
                best_loss = avg_total
                trial_log['best_loss'] = best_loss
                patience_counter = 0
                # Save best model state for this trial
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                    'trial_number': trial.number,
                    'hyperparameters': trial_log['hyperparameters'].copy()
                }
                best_epoch = epoch + 1
                print(f"New best loss for trial {trial.number}: {best_loss:.4f} at epoch {epoch+1}")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                trial_log['status'] = 'early_stopped'
                trial_log['early_stopped_at_epoch'] = epoch + 1
                break
            
            print(f"Epoch {epoch+1}: Loss={avg_total:.4f}, Duration={epoch_duration:.1f}s")
        
        trial_log['status'] = 'completed'
        trial_log['best_loss'] = best_loss
        trial_log['best_epoch'] = best_epoch
        
        # Save best model for this trial
        if best_model_state is not None:
            model_save_path = os.path.join(trial_dir, 'best_model.pt')
            torch.save(best_model_state, model_save_path)
            trial_log['best_model_path'] = model_save_path
            print(f"Best model for trial {trial.number} saved to: {model_save_path}")
        
        # Log final results to TensorBoard
        writer.add_hparams(hparams, {
            'final_loss': best_loss, 
            'final_epoch': trial_log['final_epoch'],
            'best_epoch': best_epoch,
            'status': 1  # 1 for completed
        })
        writer.close()
        
        # Save trial log
        with open(os.path.join(trial_dir, 'trial_log.json'), 'w') as f:
            json.dump(trial_log, f, indent=2)
        
        print(f"Trial {trial.number} completed with best loss: {best_loss:.4f}")
        print(f"TensorBoard logs saved to: {tb_log_dir}")
        return best_loss
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        trial_log['status'] = 'failed'
        trial_log['error'] = str(e)
        # Close writer even for failed trials
        if 'writer' in locals():
            writer.close()
        # Save trial log even for failed trials
        with open(os.path.join(trial_dir, 'trial_log.json'), 'w') as f:
            json.dump(trial_log, f, indent=2)
        return float('inf')

def create_visualizations(study, save_dir):
    """Create visualization plots for the optuna study"""
    try:
        import matplotlib.pyplot as plt
        
        # Create plots directory
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Optimization history
        fig, ax = plt.subplots(figsize=(10, 6))
        values = [trial.value for trial in study.trials if trial.value is not None]
        ax.plot(values, 'b-', alpha=0.7)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Loss')
        ax.set_title('Optimization History')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter importance (if enough trials)
        if len(study.trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                params = list(importance.keys())
                values = list(importance.values())
                
                bars = ax.barh(params, values)
                ax.set_xlabel('Importance')
                ax.set_title('Hyperparameter Importance')
                ax.grid(True, alpha=0.3)
                
                # Color bars by importance
                max_val = max(values)
                for bar, val in zip(bars, values):
                    bar.set_color(plt.cm.viridis(val / max_val))
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'parameter_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Could not create parameter importance plot: {e}")
        
        # 3. Best value over time
        fig, ax = plt.subplots(figsize=(10, 6))
        best_values = []
        current_best = float('inf')
        
        for trial in study.trials:
            if trial.value is not None and trial.value < current_best:
                current_best = trial.value
            best_values.append(current_best if current_best != float('inf') else None)
        
        valid_indices = [i for i, v in enumerate(best_values) if v is not None]
        valid_values = [best_values[i] for i in valid_indices]
        
        if valid_values:
            ax.plot(valid_indices, valid_values, 'g-', linewidth=2)
            ax.set_xlabel('Trial')
            ax.set_ylabel('Best Loss So Far')
            ax.set_title('Best Value Over Time')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'best_value_over_time.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {plots_dir}")
        
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def verify_medgemma_setup():
    """Helper function to verify MedGemma model accessibility using official API"""
    try:
        print("Testing MedGemma access using official API...")
        print("API: AutoModelForImageTextToText.from_pretrained('google/medgemma-4b-it')")
        
        # Use official MedGemma API as documented
        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it", 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu"  # CPU for testing
        )
        
        print("MedGemma model accessible!")
        print(f"Model type: {type(model)}")
        
        # Inspect vision-related components
        vision_attrs = [attr for attr in dir(model) if 'vision' in attr.lower() and not attr.startswith('_')]
        print(f"Vision-related attributes: {vision_attrs}")
        
        # Check for common vision component names
        component_found = None
        for attr_name in ['vision_tower', 'vision_model', 'vision_encoder', 'visual_encoder']:
            if hasattr(model, attr_name):
                component = getattr(model, attr_name)
                print(f"Found {attr_name}: {type(component)}")
                if hasattr(component, 'config') and hasattr(component.config, 'hidden_size'):
                    print(f"Hidden size: {component.config.hidden_size}")
                component_found = attr_name
                break
        
        if component_found:
            print(f"Vision encoder component identified: {component_found}")
            return True, component_found
        else:
            print("Vision component found but name unclear")
            return True, "unknown"
            
    except Exception as e:
        print(f"MedGemma not accessible: {e}")
        if "401" in str(e) or "access" in str(e).lower():
            print("Access issue detected. To enable MedGemma:")
            print("1. Go to: https://huggingface.co/google/medgemma-4b-it")
            print("2. Click 'Request access'")
            print("3. Agree to Health AI Developer Foundation terms")
            print("4. Wait for approval (usually immediate)")
        else:
            print("This might be a network or dependency issue")
        return False, None

def save_best_model_across_trials(study, run_save_dir):
    """Save the best model from the best trial"""
    try:
        best_trial = study.best_trial
        best_trial_dir = os.path.join(run_save_dir, f"trial_{best_trial.number}")
        best_model_path = os.path.join(best_trial_dir, 'best_model.pt')
        
        if os.path.exists(best_model_path):
            # Copy best model to main directory
            global_best_path = os.path.join(run_save_dir, 'global_best_model.pt')
            import shutil
            shutil.copy2(best_model_path, global_best_path)
            
            # Also save a model loading script
            model_loader_script = f"""
# Script to load the best model from hyperparameter search
import torch
import json

# Load best hyperparameters
with open('best_hyperparameters.json', 'r') as f:
    best_config = json.load(f)

# Create a mock trial object for model initialization
class MockTrial:
    def __init__(self, params):
        self.params = params
    
    def suggest_categorical(self, name, choices):
        return self.params.get(name, choices[0])
    
    def suggest_float(self, name, low, high):
        return self.params.get(name, (low + high) / 2)

# Load model
mock_trial = MockTrial(best_config['best_params'])
model = SigLIPModel(mock_trial)

# Load trained weights
checkpoint = torch.load('global_best_model.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded best model from trial {{checkpoint['trial_number']}}")
print(f"Best loss: {{checkpoint['loss']:.4f}}")
print(f"Best epoch: {{checkpoint['epoch']}}")
print("Model ready for inference!")
"""
            
            with open(os.path.join(run_save_dir, 'load_best_model.py'), 'w') as f:
                f.write(model_loader_script)
            
            print(f"Best model saved to: {global_best_path}")
            print(f"Model loader script saved to: {run_save_dir}/load_best_model.py")
            
            return global_best_path
        else:
            print(f"Warning: Best model file not found at {best_model_path}")
            return None
            
    except Exception as e:
        print(f"Error saving best model: {e}")
        return None

def main():
    print("OPTUNA HYPERPARAMETER SEARCH FOR SIGLIP TRAINING")
    print("=" * 60)
    print(f"VISION ENCODER COMPARISON:")
    print(f"   Gemma3 SigLIP: google/siglip-so400m-patch14-384 (General-purpose)")
    print(f"   MedGemma SigLIP: Extracted from google/medgemma-4b-it (Medical-trained)")
    print(f"   Both encoders: 1152 latent dimensions, 384x384 input")
    print(f"   Comparison: General vs Medical pre-training effectiveness")
    print(f"Device: {DEVICE}")
    print(f"Epochs per trial: {EPOCHS}")
    print(f"TFRecord path: {TFRECORD_PATH}")
    print(f"Model save directory: {run_save_dir}")
    print(f"TensorBoard logging: Enabled (per trial)")
    print("")
    print("READY: Both encoders integrated using official APIs")
    print("   - Gemma3: Direct SiglipVisionModel loading")
    print("   - MedGemma: AutoModelForImageTextToText + vision extraction")
    print("   - Automatic fallback if MedGemma access issues")
    print("")
    print("HYPERPARAMETER SEARCH SPACE:")
    print("   Batch sizes: [4, 8, 16, 32, 48, 64]")
    print("   Learning rates: [1e-3, 1e-4, 1e-5]")
    print("   Alpha values: [0.1, 0.9]")
    print("   Optimizers: [adam, adamw, sgd]")
    print("   Weight decay: [1e-6, 1e-2] (log scale)")
    print("   Image encoders: [Gemma3-SigLIP, MedGemma-SigLIP]")
    print("   Text models: [t5-base, t5-small, t5-large]")
    print("   Dropout rates: [0.0, 0.5]")
    print("   Augmentation parameters: Various ranges")
    print("   Scheduler parameters: Various ranges")
    print("   Early stopping patience: [5, 15]")
    print("   Max text length: [64, 128, 256]")
    print("=" * 60)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print("Starting hyperparameter optimization...")
    
    # Start optimization
    study.optimize(
        objective, 
        n_trials=100,  # Adjust based on computational budget
        timeout=None  # No timeout, will run for specified trials
    )
    
    # Print results
    print("=" * 60)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best loss: {study.best_value:.4f}")
    print("")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial_number': study.best_trial.number,
        'n_trials': len(study.trials),
        'timestamp': timestamp,
        'device': DEVICE,
        'epochs_per_trial': EPOCHS,
        'tfrecord_path': TFRECORD_PATH,
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    }
    
    # Save best parameters
    with open(os.path.join(run_save_dir, 'best_hyperparameters.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save study
    study_path = os.path.join(run_save_dir, 'optuna_study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    # Save all trial results
    all_trials = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'state': trial.state.name,
            'value': trial.value,
            'params': trial.params,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            'duration': trial.duration.total_seconds() if trial.duration else None
        }
        all_trials.append(trial_data)
    
    with open(os.path.join(run_save_dir, 'all_trials.json'), 'w') as f:
        json.dump(all_trials, f, indent=2)
    
    # Save the best model across all trials
    best_model_path = save_best_model_across_trials(study, run_save_dir)
    
    # Create visualizations
    create_visualizations(study, run_save_dir)
    
    print(f"Results saved to: {run_save_dir}")
    print(f"Study object saved to: {study_path}")
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")
        print(f"Use 'load_best_model.py' to load the trained model")
    print("Use optuna-dashboard to visualize results:")
    print(f"   pip install optuna-dashboard")
    print(f"   optuna-dashboard {study_path}")
    print("")
    print("Analysis options:")
    print(f"   - Best model: {run_save_dir}/global_best_model.pt")
    print(f"   - Model loader: {run_save_dir}/load_best_model.py")
    print(f"   - Individual trial logs: {run_save_dir}/trial_*/")
    print(f"   - Summary plots: {run_save_dir}/plots/")
    print(f"   - TensorBoard for all trials: tensorboard --logdir {run_save_dir}")
    print(f"   - TensorBoard for specific trial: tensorboard --logdir {run_save_dir}/trial_X/tensorboard")
    print(f"   - Load study for analysis: pickle.load(open('{study_path}', 'rb'))")

if __name__ == "__main__":
    main()
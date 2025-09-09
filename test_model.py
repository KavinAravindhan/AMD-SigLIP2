import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
import glob
from datetime import datetime
from embedder import TextEmbedder
from alignment import SigLIPLoss

# nohup python test_model.py > /home/kavin/AMD-SigLIP2/terminal_output_nohup/test_model_3.txt 2>&1 &

# Test dataset paths
TEST_DATASET_PATH = "/media/16TB_Storage/kavin/amd_siglip/OCT_test_dataset"
NORMAL_IMAGES_PATH = os.path.join(TEST_DATASET_PATH, "normal3Channel")
WET_AMD_IMAGES_PATH = os.path.join(TEST_DATASET_PATH, "wetAMD3Channel")

# Best model path
# BEST_MODEL_PATH = "/media/16TB_Storage/kavin/amd_siglip/saved_models/optuna_run_20250826_211533/global_best_model.pt"
# BEST_MODEL_PATH = "/media/16TB_Storage/michael/models/kevin_amd/optuna_run_20250807_143750/global_best_model.pt"
BEST_MODEL_PATH = "/media/16TB_Storage/kavin/amd_siglip/saved_models/VQA_v4_1_caption/best_model.pt"

# Results save path with timestamp
RESULTS_BASE_PATH = "/media/16TB_Storage/kavin/amd_siglip/test_model"

# Best hyperparameters (from your Optuna results)
BEST_PARAMS = {
    'batch_size': 8,
    'learning_rate': 0.0001,
    'alpha': 0.8840963962895334,
    'optimizer': 'adamw',
    'weight_decay': 1.6079555710533247e-06,
    'image_encoder': 'google/siglip-so400m-patch14-384',
    'text_model': 'google-t5/t5-base',
    'dropout_rate': 0.057129660535791646
}

# BEST_PARAMS = {
#     'batch_size': 16,
#     'learning_rate': 0.0001,
#     'alpha': 0.8993188484302025,
#     'optimizer': 'adamw',
#     'weight_decay': 2.0972034386280457e-06,
#     'image_encoder': 'google/siglip-so400m-patch14-384',
#     'text_model': 'google-t5/t5-small',
#     'dropout_rate': 0.11538216057289494
# }

# Constants
MAX_TEXT_LEN = 128
IMAGE_SIZE = 384  # SigLIP input size

# Device setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

class SigLIPModel(nn.Module):
    """SigLIP model for AMD classification"""
    def __init__(self, dropout_rate=0.057129660535791646):
        super().__init__()
        
        # Load Gemma3 SigLIP encoder (from best hyperparameters)
        print("Loading Gemma3 SigLIP encoder...")
        self.image_encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        encoder_output_dim = 1152
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.cls_head = nn.Linear(encoder_output_dim, 2)  # 2 classes (normal/wet)
        
        # SigLIP loss (needed for model structure)
        self.siglip_loss = SigLIPLoss(
            latent_dim=encoder_output_dim,
            text_model="google-t5/t5-base",
            # text_model="google-t5/t5-small",
            max_txt_len=MAX_TEXT_LEN,
            pool="mean",
            dtype=torch.float32
        )
        
    def forward(self, images, input_ids=None, attention_mask=None):
        """Forward pass for inference"""
        # Extract features from image encoder
        img_features = self.image_encoder(pixel_values=images).last_hidden_state  # (B, 729, 1152)
        
        # Classification: Use CLS token (first token)
        cls_features = self.dropout(img_features[:, 0])  # (B, 1152)
        cls_logits = self.cls_head(cls_features)  # (B, 2)
        
        return cls_logits

def load_best_model():
    """Load the best model from hyperparameter search"""
    print("Loading best trained model...")
    
    # Create model with best hyperparameters
    model = SigLIPModel(dropout_rate=BEST_PARAMS['dropout_rate'])
    
    # Load trained weights
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model file not found: {BEST_MODEL_PATH}")
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print("Model loaded successfully!")
    # print(f"   Trial number: {checkpoint['trial_number']}")
    print(f"   Training loss: {checkpoint['loss']:.4f}")
    print(f"   Best epoch: {checkpoint['epoch']}")
    
    return model, checkpoint

def preprocess_image(image_path):
    """Preprocess image to match training preprocessing"""
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Resize to 384x384 (SigLIP input size)
        img_resized = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Normalize to [-1, 1] for SigLIP
        img_tensor = (img_tensor - 0.5) / 0.5
        
        return img_tensor
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def load_test_dataset():
    """Load test images and labels"""
    print("Loading test dataset...")
    
    # Find all image files
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
    
    normal_images = []
    for ext in image_extensions:
        normal_images.extend(glob.glob(os.path.join(NORMAL_IMAGES_PATH, ext)))
    
    wet_amd_images = []
    for ext in image_extensions:
        wet_amd_images.extend(glob.glob(os.path.join(WET_AMD_IMAGES_PATH, ext)))
    
    print(f"Found {len(normal_images)} normal images")
    print(f"Found {len(wet_amd_images)} wet AMD images")
    
    if len(normal_images) == 0:
        print(f"Warning: No normal images found in {NORMAL_IMAGES_PATH}")
    if len(wet_amd_images) == 0:
        print(f"Warning: No wet AMD images found in {WET_AMD_IMAGES_PATH}")
    
    # Create dataset
    test_data = []
    
    # Add normal images (label = 0)
    for img_path in normal_images:
        test_data.append((img_path, 0, "normal"))
    
    # Add wet AMD images (label = 1)
    for img_path in wet_amd_images:
        test_data.append((img_path, 1, "wet_amd"))
    
    print(f"Total test images: {len(test_data)}")
    return test_data

def evaluate_model(model, test_data, batch_size=8):
    """Evaluate model on test dataset"""
    print("Starting model evaluation...")
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    failed_images = []
    
    model.eval()
    
    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
            batch_data = test_data[i:i+batch_size]
            
            # Prepare batch
            batch_images = []
            batch_labels = []
            batch_paths = []
            
            for img_path, label, class_name in batch_data:
                img_tensor = preprocess_image(img_path)
                if img_tensor is not None:
                    batch_images.append(img_tensor)
                    batch_labels.append(label)
                    batch_paths.append(img_path)
                else:
                    failed_images.append(img_path)
            
            if len(batch_images) == 0:
                continue
            
            # Stack batch and move to device
            images = torch.stack(batch_images).to(DEVICE)
            labels = torch.tensor(batch_labels).to(DEVICE)
            
            # Forward pass (no text inputs needed for classification)
            try:
                logits = model(images)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                print(f"Error during inference for batch starting at index {i}: {e}")
                failed_images.extend(batch_paths)
    
    if failed_images:
        print(f"Warning: {len(failed_images)} images failed to process")
        if len(failed_images) <= 5:
            print("Failed images:", failed_images)
        else:
            print("First 5 failed images:", failed_images[:5])
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def calculate_metrics(predictions, labels, probabilities):
    """Calculate comprehensive evaluation metrics"""
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Basic accuracy
    accuracy = (predictions == labels).mean()
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Class-wise accuracy
    normal_mask = labels == 0
    wet_amd_mask = labels == 1
    
    normal_accuracy = (predictions[normal_mask] == labels[normal_mask]).mean() if normal_mask.sum() > 0 else 0
    wet_amd_accuracy = (predictions[wet_amd_mask] == labels[wet_amd_mask]).mean() if wet_amd_mask.sum() > 0 else 0
    
    print(f"Normal OCT Accuracy: {normal_accuracy:.4f} ({normal_accuracy*100:.2f}%)")
    print(f"Wet AMD Accuracy: {wet_amd_accuracy:.4f} ({wet_amd_accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                Normal  Wet AMD")
    print(f"Actual Normal    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       Wet AMD   {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Sensitivity (Recall for Wet AMD)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Specificity (True Negative Rate for Normal)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # Precision for Wet AMD
    precision_wet = tp / (tp + fp) if (tp + fp) > 0 else 0
    # F1 score for Wet AMD
    f1_wet = 2 * (precision_wet * sensitivity) / (precision_wet + sensitivity) if (precision_wet + sensitivity) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"Sensitivity (Wet AMD Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"Specificity (Normal Recall):  {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"Precision (Wet AMD):          {precision_wet:.4f} ({precision_wet*100:.2f}%)")
    print(f"F1 Score (Wet AMD):           {f1_wet:.4f}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    class_names = ['Normal', 'Wet AMD']
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    print(report)
    
    # ROC AUC
    try:
        auc_score = roc_auc_score(labels, probabilities[:, 1])  # Wet AMD probabilities
        print(f"ROC AUC Score: {auc_score:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")
        auc_score = None
    
    # Dataset statistics
    print(f"\nDataset Statistics:")
    print(f"   Total samples: {len(labels)}")
    print(f"   Normal samples: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"   Wet AMD samples: {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'normal_accuracy': normal_accuracy,
        'wet_amd_accuracy': wet_amd_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision_wet': precision_wet,
        'f1_wet': f1_wet,
        'confusion_matrix': cm,
        'classification_report': report,
        'auc_score': auc_score
    }

def create_visualizations(predictions, labels, probabilities, save_dir):
    """Create visualization plots for evaluation results"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(labels, predictions)
        
        # Create percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create combined annotations (count and percentage)
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=['Normal', 'Wet AMD'], 
                   yticklabels=['Normal', 'Wet AMD'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix\n(Count and Percentage)', fontsize=16, pad=20)
        plt.ylabel('Actual Class', fontsize=14)
        plt.xlabel('Predicted Class', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        if len(probabilities.shape) == 2 and probabilities.shape[1] == 2:
            try:
                fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
                auc_score = roc_auc_score(labels, probabilities[:, 1])
                
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC Curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
                plt.xlabel('False Positive Rate', fontsize=14)
                plt.ylabel('True Positive Rate', fontsize=14)
                plt.title('ROC Curve - Wet AMD Detection', fontsize=16, pad=20)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Could not create ROC curve: {e}")
        
        # 3. Class-wise Accuracy Bar Plot
        normal_accuracy = (predictions[labels == 0] == labels[labels == 0]).mean() if (labels == 0).sum() > 0 else 0
        wet_amd_accuracy = (predictions[labels == 1] == labels[labels == 1]).mean() if (labels == 1).sum() > 0 else 0
        
        plt.figure(figsize=(10, 8))
        classes = ['Normal', 'Wet AMD']
        accuracies = [normal_accuracy, wet_amd_accuracy]
        colors = ['lightblue', 'lightcoral']
        
        bars = plt.bar(classes, accuracies, color=colors, edgecolor='black', alpha=0.8, width=0.6)
        plt.ylim(0, 1.1)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title('Class-wise Accuracy', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add accuracy text on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}\n({acc*100:.1f}%)', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_accuracies.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Prediction Confidence Distribution
        plt.figure(figsize=(15, 5))
        
        # Overall confidence distribution
        plt.subplot(1, 3, 1)
        max_confidences = np.max(probabilities, axis=1)
        plt.hist(max_confidences, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Overall Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        # Confidence for correct predictions
        plt.subplot(1, 3, 2)
        correct_mask = predictions == labels
        if correct_mask.sum() > 0:
            correct_confidences = np.max(probabilities[correct_mask], axis=1)
            plt.hist(correct_confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Correct Predictions')
        plt.grid(True, alpha=0.3)
        
        # Confidence for incorrect predictions
        plt.subplot(1, 3, 3)
        incorrect_mask = predictions != labels
        if incorrect_mask.sum() > 0:
            incorrect_confidences = np.max(probabilities[incorrect_mask], axis=1)
            plt.hist(incorrect_confidences, bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Incorrect Predictions')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {save_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def analyze_misclassifications(test_data, predictions, labels, probabilities, save_dir, max_examples=20):
    """Analyze and save examples of misclassified images"""
    try:
        misclassified_dir = os.path.join(save_dir, 'misclassified_examples')
        os.makedirs(misclassified_dir, exist_ok=True)
        
        # Find misclassified images
        misclassified_indices = np.where(predictions != labels)[0]
        
        if len(misclassified_indices) == 0:
            print("Perfect classification! No misclassified images to analyze.")
            return
        
        print(f"\nAnalyzing {len(misclassified_indices)} misclassified images...")
        print(f"Saving up to {max_examples} examples...")
        
        # Sort by confidence (most confident wrong predictions first)
        misclassified_confidences = np.max(probabilities[misclassified_indices], axis=1)
        sorted_indices = misclassified_indices[np.argsort(-misclassified_confidences)]
        
        # Sample examples
        sample_indices = sorted_indices[:max_examples]
        
        misclassification_log = []
        
        for idx in sample_indices:
            img_path, true_label, true_class = test_data[idx]
            pred_label = predictions[idx]
            pred_class = "normal" if pred_label == 0 else "wet_amd"
            confidence = np.max(probabilities[idx])
            
            # Copy image to misclassified directory with descriptive name
            img_name = os.path.basename(img_path)
            name, ext = os.path.splitext(img_name)
            new_name = f"{name}_true_{true_class}_pred_{pred_class}_conf_{confidence:.3f}{ext}"
            
            import shutil
            shutil.copy2(img_path, os.path.join(misclassified_dir, new_name))
            
            misclassification_log.append({
                'original_path': img_path,
                'true_label': int(true_label),
                'true_class': true_class,
                'predicted_label': int(pred_label),
                'predicted_class': pred_class,
                'confidence': float(confidence),
                'saved_as': new_name
            })
        
        # Save misclassification log
        with open(os.path.join(misclassified_dir, 'misclassification_analysis.json'), 'w') as f:
            json.dump(misclassification_log, f, indent=2)
        
        print(f"Misclassified examples saved to: {misclassified_dir}")
        print(f"   - {len(sample_indices)} example images copied")
        print(f"   - Analysis saved to misclassification_analysis.json")
        
    except Exception as e:
        print(f"Error analyzing misclassifications: {e}")

def save_evaluation_summary(metrics, checkpoint_info, test_data_info, save_dir):
    """Save comprehensive evaluation summary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'evaluation_timestamp': timestamp,
        'model_info': {
            'model_path': BEST_MODEL_PATH,
            'trial_number': checkpoint_info.get('trial_number'),
            'training_loss': float(checkpoint_info.get('loss', 0)),
            'best_epoch': checkpoint_info.get('epoch'),
            'hyperparameters': BEST_PARAMS
        },
        'test_dataset_info': test_data_info,
        'evaluation_metrics': {
            'overall_accuracy': float(metrics['accuracy']),
            'normal_accuracy': float(metrics['normal_accuracy']),
            'wet_amd_accuracy': float(metrics['wet_amd_accuracy']),
            'sensitivity': float(metrics['sensitivity']),
            'specificity': float(metrics['specificity']),
            'precision_wet_amd': float(metrics['precision_wet']),
            'f1_score_wet_amd': float(metrics['f1_wet']),
            'roc_auc': float(metrics['auc_score']) if metrics['auc_score'] is not None else None
        },
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    # Save summary
    with open(os.path.join(save_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation summary saved to: {os.path.join(save_dir, 'evaluation_summary.json')}")

def create_timestamped_results_dir():
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_BASE_PATH, f"evaluation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def main():
    print(f"Test dataset: {TEST_DATASET_PATH}")
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BEST_PARAMS['batch_size']}")
    print("")
    
    # Verify paths exist
    required_paths = [TEST_DATASET_PATH, NORMAL_IMAGES_PATH, WET_AMD_IMAGES_PATH, BEST_MODEL_PATH]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")
    
    # Create timestamped results directory
    results_dir = create_timestamped_results_dir()
    print(f"Results will be saved to: {results_dir}")
    print("")
    
    # Load model
    model, checkpoint_info = load_best_model()
    
    # Load test dataset
    test_data = load_test_dataset()
    
    if len(test_data) == 0:
        raise ValueError("No test images found!")
    
    # Evaluate model
    predictions, labels, probabilities = evaluate_model(
        model, test_data, batch_size=BEST_PARAMS['batch_size']
    )
    
    if len(predictions) == 0:
        raise ValueError("No successful predictions! Check image preprocessing.")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels, probabilities)
    
    # Test dataset info
    test_data_info = {
        'total_images': len(test_data),
        'normal_images': len([d for d in test_data if d[1] == 0]),
        'wet_amd_images': len([d for d in test_data if d[1] == 1]),
        'successful_predictions': len(predictions),
        'normal_path': NORMAL_IMAGES_PATH,
        'wet_amd_path': WET_AMD_IMAGES_PATH
    }
    
    # Save comprehensive results
    save_evaluation_summary(metrics, checkpoint_info, test_data_info, results_dir)
    
    # Create visualizations
    create_visualizations(predictions, labels, probabilities, results_dir)
    
    # Analyze misclassifications
    analyze_misclassifications(test_data, predictions, labels, probabilities, results_dir)
    
    # Final summary
    print(f"\nResults summary:")
    print(f"   Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Normal Accuracy:  {metrics['normal_accuracy']:.4f} ({metrics['normal_accuracy']*100:.2f}%)")
    print(f"   Wet AMD Accuracy: {metrics['wet_amd_accuracy']:.4f} ({metrics['wet_amd_accuracy']*100:.2f}%)")
    print(f"   Sensitivity:      {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"   Specificity:      {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    if metrics['auc_score'] is not None:
        print(f"   ROC AUC:          {metrics['auc_score']:.4f}")
    
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    main()
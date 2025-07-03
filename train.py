import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tfrecord.torch.dataset import TFRecordDataset
from transformers import AutoProcessor, AutoModel
from PIL import Image
import io
import torch.nn.functional as F
from embedder import TextEmbedder

# ───────────── Config ──────────────
TFRECORD_PATH = "dataset.tfrecord"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_TEXT_LEN = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────── Dataset Loader ──────────────
description = {
    "image_raw": "byte",
    "label": "int",
    "transcript": "byte",
}

def collate_fn(batch):
    images, labels, transcripts = [], [], []
    for item in batch:
        img = Image.open(io.BytesIO(item["image_raw"])).convert("RGB")
        img = TF.resize(img, [224, 224])
        img = TF.to_tensor(img)
        images.append(img)
        labels.append(item["label"])
        transcripts.append(item["transcript"].decode())  # string
    return torch.stack(images), torch.tensor(labels), transcripts

dataset = TFRecordDataset(TFRECORD_PATH, None, description)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ───────────── Load Models ──────────────
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
image_encoder = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(DEVICE)
text_embedder = TextEmbedder(model_name="google-t5/t5-base", max_len=MAX_TEXT_LEN).to(DEVICE)

# ───────────── Decoder Model ──────────────
class SigLIP2CoCa(nn.Module):
    def __init__(self, image_encoder, dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.cls_head = nn.Linear(dim, 2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, pixel_values, encoded_text):
        image_out = self.image_encoder(pixel_values=pixel_values).last_hidden_state  # (B,N,D)
        img_cls = image_out[:, 0]  # CLS token
        cls_logits = self.cls_head(img_cls)

        memory = image_out.permute(1, 0, 2)      # (N, B, D)
        tgt = encoded_text.permute(1, 0, 2)      # (T, B, D)
        dec_out = self.decoder(tgt, memory)      # (T, B, D)
        return cls_logits, dec_out.permute(1, 0, 2)  # (B, D), (B, T, D)

model = SigLIP2CoCa(image_encoder).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ───────────── Training ──────────────
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_cls, total_align = 0, 0, 0
    for imgs, labels, transcripts in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Temporarily use label names as transcripts
        text_inputs = ["AMD" if l == 1 else "Normal" for l in labels]
        ids_np, mask_np = text_embedder.tokenize(text_inputs)
        input_ids = torch.tensor(ids_np).to(DEVICE)
        attn_mask = torch.tensor(mask_np).bool().to(DEVICE)

        with torch.no_grad():
            encoded_text = text_embedder.encode(input_ids, attn_mask)  # (B, L, D)

        # Prepare image input
        pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(DEVICE)
        cls_logits, dec_out = model(pixel_values, encoded_text)

        # Losses
        loss_cls = F.cross_entropy(cls_logits, labels)
        loss_align = F.mse_loss(dec_out, encoded_text)
        loss = loss_cls + 0.5 * loss_align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_align += loss_align.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {avg_loss:.4f} | Cls Loss: {total_cls/len(dataloader):.4f} | Align Loss: {total_align/len(dataloader):.4f}")

# Save model and optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'siglip2_model_final.pt')

print("Model saved as 'siglip2_model_final.pt'")

# import os
# os.makedirs("checkpoints", exist_ok=True)
# torch.save(model.state_dict(), "checkpoints/siglip2_coca_final.pth")

# checkpoint = torch.load('siglip2_model_final.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# model.eval()
import io
from tkinter import Image
import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
from tfrecord.torch.dataset import TFRecordDataset
import torch.nn.functional as F

# Dataset utils
description = {
  "image_raw": "byte",
  "label": "int",
  "transcript": "byte",
}
def collate_fn(batch):
    imgs, labels, trans = zip(*[(item["image_raw"], item["label"], item["transcript"]) for item in batch])
    imgs = [torch.from_numpy(np.array(Image.open(io.BytesIO(i))).transpose(2,0,1))/255.0 for i in imgs]
    labels = torch.tensor(labels)
    # map transcripts "A"/"N" to token 0/1
    trans = torch.tensor([1 if t.decode()=='A' else 0 for t in trans])
    return torch.stack(imgs), labels, trans

# TFRecord dataset
ds = TFRecordDataset("dataset.tfrecord", None, description)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Model
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
img_encoder = AutoModel.from_pretrained("google/siglip2-base-patch16-224")

class SigLIP2CoCa(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = img_encoder
        self.token_emb = nn.Embedding(2, 512)
        dec_layer = nn.TransformerDecoderLayer(512, 8)
        self.decoder = nn.TransformerDecoder(dec_layer, 2)
        self.cls_head = nn.Linear(512, 2)
    def forward(self, pixel_values, tgt_tokens):
        out = self.enc(pixel_values=pixel_values).last_hidden_state
        cls_token = out[:,0]
        cls_logits = self.cls_head(cls_token)
        tgt_emb = self.token_emb(tgt_tokens).permute(1,0,2)
        memory = out.permute(1,0,2)
        dec = self.decoder(tgt_emb, memory).permute(1,0,2)
        return cls_logits, dec

model = SigLIP2CoCa().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training
for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels, trans in loader:
        imgs = imgs.cuda()
        labels = labels.cuda()
        trans = trans.cuda().unsqueeze(1).repeat(1,1)
        pix = processor(images=imgs, return_tensors="pt").pixel_values.cuda()
        cls_logits, dec_out = model(pix, trans)

        loss_cls = F.cross_entropy(cls_logits, labels)
        loss_align = F.cross_entropy(dec_out.view(-1,512), trans.view(-1))
        loss = loss_cls + 0.5 * loss_align

        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: avg_loss={total_loss/len(loader):.4f}")

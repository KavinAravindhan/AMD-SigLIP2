import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from renaanalysis.text.embedder import TextEmbedder


class SigLIPLoss(nn.Module):
    """
    SigLIP contrastive loss (Baek et al., ’23).

    • Accepts **text token ids / masks** and **EEG latents that have
      already been cross-attended with text**.
    • Computes a *symmetric* loss: EEG→TXT and TXT→EEG.
    • Learns a temperature `t = exp(t_prime)` and bias `b` as in the paper.

    Args
    ----
    latent_dim     : channel dim of EEG latents after X-Attn
    text_model     : HF name of the frozen text encoder (defaults to T5-base)
    pool           : "mean" ‖ "cls"  (how to pool EEG tokens)
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        text_model: str = "google-t5/t5-base",
        max_txt_len: int = 128,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
        pool: str = "mean",
    ):
        super().__init__()
        self.text = TextEmbedder(
            model_name=text_model,
            max_len=max_txt_len,
            dtype=dtype,
        )
        self.pool = pool.lower()
        txt_dim = self.text.encoder.config.d_model

        # If dimensions differ, project text → latent_dim
        self.txt_proj = (
            nn.Identity()
            if txt_dim == latent_dim
            else nn.Linear(txt_dim, latent_dim, bias=False)
        )

        # learnable temperature + bias (initialise as in SigLIP impl)
        self.t_prime = nn.Parameter(torch.tensor(0.07).log())  # so exp ≈ 0.07
        self.bias    = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    @staticmethod
    def _masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)                     # (B,L,1)
        denom = mask.sum(dim=1).clamp(min=1)          # avoid ÷0
        return (seq * mask).sum(dim=1) / denom        # (B,D)

    # ------------------------------------------------------------------
    def forward(
        self,
        physio_latent_tokens: torch.Tensor,   # (B, N_lat, D_lat) – after X-Attn
        text_ids:           torch.LongTensor,     # (B, L_txt)
        text_mask:          torch.BoolTensor,     # (B, L_txt)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        loss  : scalar
        logits: (B,B) cosine-similarity matrix (optional debugging)
        """
        # ---- Text side --------------------------------------------------
        with torch.no_grad():
            txt_seq = self.text.encode(text_ids, text_mask.bool())  # (B,L,D_txt)
        z_txt = self._masked_mean(txt_seq, text_mask.to(dtype=physio_latent_tokens.dtype))     # (B,D_txt)
        z_txt = F.normalize(self.txt_proj(z_txt), dim=-1)      # (B,D_lat)

        # ---- EEG side ---------------------------------------------------
        if self.pool == "mean":
            z_physio = F.normalize(physio_latent_tokens.mean(dim=1), dim=-1)
        elif self.pool == "cls":
            z_physio = F.normalize(physio_latent_tokens[:, 0], dim=-1)
        else:
            raise ValueError("pool must be 'mean' or 'cls'")

        # 3) ---- logits --------------------------------------------------
        t   = self.t_prime.exp()
        log = z_physio @ z_txt.t() * t + self.bias          # (B,B)

        B   = log.size(0)
        lbl = torch.eye(B, device=log.device).mul_(2).sub_(1)  # +1 diag, -1 off

        pair_loss = -F.logsigmoid(lbl * log)                   # (B,B)

        # -----------------------------------------------------------------
        # per-sample loss = average of its *row* and its *column*
        # -----------------------------------------------------------------
        row_loss = pair_loss.mean(dim=1)   # each EEG as query over texts
        col_loss = pair_loss.mean(dim=0)   # each text as query over EEGs
        loss_batch = 0.5 * (row_loss + col_loss)               # (B,)

        loss_mean = loss_batch.mean()      # scalar (same value as before)

        return loss_mean, loss_batch.detach(), log.detach()

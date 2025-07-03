from typing import List
from typing import Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel


class TextEmbedder(nn.Module):
    """
    Minimal wrapper around a *frozen* T5-base (or any seq-to-seq encoder).

    ────────────  Usage  ─────────────
    >>> txt = TextEmbedder("google-t5/t5-base", max_len=128)
    >>> ids, mask = txt.tokenize("some text")
    >>> z_txt    = txt.encode(torch.tensor(ids)[None, :],
                              torch.tensor(mask)[None, :])
    """
    def __init__(
        self,
        model_name: str = "google-t5/t5-base",
        max_len: int = 128,
        dtype: torch.dtype = torch.float16,   # keeps weights ≈2× smaller
    ):
        super(TextEmbedder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = T5EncoderModel.from_pretrained(model_name).to(dtype=dtype)
        self.encoder.eval()
        for p in self.encoder.parameters():   # freeze
            p.requires_grad = False

        self.max_len = max_len

    # ------------------------------------------------------------ tokenize
    @torch.no_grad()
    def tokenize(self, text: str | List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        padded position are set to 0, and attention mask to 1.
        For example, if max_len=5 and text="hellow world",
        the tokenizer will return:
        input_ids = [21820, 296,  0,  0,  0]
        attention_mask = [1, 1, 0, 0, 0]

        Returns  (input_ids[int32], attn_mask[int8])  shaped (max_len,)
        (If *text* is a list, shapes are (B,max_len).)
        """
        batch = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
        )
        ids   = np.asarray(batch["input_ids"],   dtype=np.int32)
        mask  = np.asarray(batch["attention_mask"], dtype=np.int8)
        return ids, mask

    # ------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.LongTensor,        # (B,L)
        attention_mask: torch.BoolTensor,   # (B,L)
    ) -> torch.Tensor:                      # (B,L,H)
        """
        Run the frozen encoder and return the sequence embeddings
        (no pooling – do that in your SigLIP loss if you wish).
        """
        input_ids     = input_ids
        attention_mask = attention_mask

        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           return_dict=True)
        return out.last_hidden_state        # (B,L,hidden)




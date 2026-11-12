"""Encoder GRU bidireccional con atención aditiva (o estado final) y cabeza lineal 256 -> 1."""

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from jailbreak_detection.models.data import PAD_IDX


class AdditiveAttention(nn.Module):
    """score_t = v^T tanh(W h_t + b); softmax sobre los tokens reales; suma ponderada."""

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, states: torch.Tensor, mask: torch.Tensor):
        scores = self.v(torch.tanh(self.proj(states))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))           # padding -> peso 0
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), states).squeeze(1), weights


class GRUEncoderClassifier(nn.Module):
    """Embeddings -> dropout -> BiGRU (1 capa) -> vector de 2*hidden -> dropout -> Linear(., 1).

    embedding_matrix=None indica que la entrada ya son vectores (ELMo, (B, T, input_size)).
    Con atención el vector es la suma ponderada de los estados de todos los tokens; sin
    atención es [h_forward_final; h_backward_final], del último token real gracias al packing."""

    def __init__(self, input_size: int, hidden_size: int = 128, use_attention: bool = True,
                 dropout: float = 0.3, embedding_matrix: np.ndarray | None = None,
                 freeze_embeddings: bool = True, attn_dim: int = 128):
        super().__init__()
        self.embedding = None
        if embedding_matrix is not None:
            if embedding_matrix.shape[1] != input_size:
                raise ValueError("input_size no coincide con la dimensión del embedding.")
            self.embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(embedding_matrix, dtype=torch.float32),
                freeze=freeze_embeddings, padding_idx=PAD_IDX)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True,
                          bidirectional=True)
        self.attention = AdditiveAttention(2 * hidden_size, attn_dim) if use_attention else None
        self.out_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, 1)
        self.output_dim = 2 * hidden_size

    def encode(self, inputs: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        """Vector del prompt (B, 256), sin pasar por la cabeza. Con return_attention=True
        devuelve también los pesos (B, T) (None si el modelo no tiene atención)."""
        x = self.embedding(inputs) if self.embedding is not None else inputs
        x = self.emb_dropout(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, h_n = self.gru(packed)
        weights = None
        if self.attention is not None:
            states, _ = pad_packed_sequence(outputs, batch_first=True, total_length=x.size(1))
            mask = (torch.arange(x.size(1), device=x.device)[None, :]
                    < lengths.to(x.device)[:, None])
            vector, weights = self.attention(states, mask)
        else:
            vector = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (fwd, bwd) de la única capa
        return (vector, weights) if return_attention else vector

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        vector, weights = self.encode(inputs, lengths, return_attention=True)
        logits = self.head(self.out_dropout(vector)).squeeze(-1)
        return (logits, weights) if return_attention else logits

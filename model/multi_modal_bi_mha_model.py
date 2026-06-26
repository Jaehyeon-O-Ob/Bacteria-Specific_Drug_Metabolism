"""
Multi-Modal Bidirectional MHA Model
Drug-Protein interaction prediction using bidirectional cross attention
with KEGG pathway features and initial drug representation.

Two cross attention paths:
  Cross1: Protein (Q) attends to Drug (K, V)  -> protein context vector
  Cross2: Drug    (Q) attends to Protein (K, V) -> drug context vector

Classifier input: prot_rep + drug_rep + kegg_rep + init_drug_rep

KEGG data source: dict mapping strain -> kegg feature vector
"""

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, matthews_corrcoef)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import xgboost as xgb


def get_activation_fn(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'swish':
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class FeedForward(nn.Module):
    def __init__(self, d_dim, output_dim, ff_dim, dropout, ff_act_fn):
        super().__init__()
        ff_fn = get_activation_fn(ff_act_fn)
        self.net = nn.Sequential(
            nn.Linear(d_dim, ff_dim),
            ff_fn,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class multi_modal_bi_cross_mha(nn.Module):
    """
    Bidirectional cross attention with KEGG modality fusion.

    Cross1: Protein -> Q, Drug -> K, V
    Cross2: Drug    -> Q, Protein -> K, V

    classifier input = prot_rep + drug_rep + kegg_rep + init_drug_rep (4 * latent_dim)

    Args:
        protein_dim         : dimension of protein embeddings
        drug_dim            : dimension of drug embeddings
        kegg_dim            : dimension of KEGG pathway features
        latent_dim          : internal hidden dimension
        cross1_mha_num_heads: heads for Cross1
        cross2_mha_num_heads: heads for Cross2
        ffc_hidden_dims     : hidden dim of classification FFN
        ff_act_fn           : activation function name
        dropout             : dropout rate
    """

    def __init__(self, protein_dim, drug_dim, kegg_dim, latent_dim,
                 cross1_mha_num_heads, cross2_mha_num_heads,
                 ffc_hidden_dims, ff_act_fn, dropout):
        super().__init__()

        assert latent_dim % cross1_mha_num_heads == 0
        assert latent_dim % cross2_mha_num_heads == 0

        self.latent_dim = latent_dim


        # ----- Cross1: Protein (Q) <- Drug (K, V) -----
        self.cross1_mha_num_heads = cross1_mha_num_heads
        self.cross1_head_dim = latent_dim // cross1_mha_num_heads

        self.prot_proj = nn.Linear(protein_dim, latent_dim)
        self.drug_proj = nn.Linear(drug_dim, latent_dim)

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

        self.prot_q = nn.Linear(latent_dim, latent_dim)
        self.drug_k = nn.Linear(latent_dim, latent_dim)
        self.drug_v = nn.Linear(latent_dim, latent_dim)

        self.o_linear1 = nn.Linear(latent_dim, latent_dim)
        self.gated_linear1 = nn.Linear(latent_dim, latent_dim)

        self.norm3 = nn.LayerNorm(latent_dim)
        self.outnorm1 = nn.LayerNorm(latent_dim)

        self.ffn1 = FeedForward(d_dim=latent_dim, ff_dim=latent_dim,
                                output_dim=latent_dim, dropout=dropout,
                                ff_act_fn=ff_act_fn)

        # ----- Cross2: Drug (Q) <- Protein (K, V) -----
        self.cross2_mha_num_heads = cross2_mha_num_heads
        self.cross2_head_dim = latent_dim // cross2_mha_num_heads

        self.norm4 = nn.LayerNorm(latent_dim)
        self.norm5 = nn.LayerNorm(latent_dim)

        self.drug_q = nn.Linear(latent_dim, latent_dim)
        self.prot_k = nn.Linear(latent_dim, latent_dim)
        self.prot_v = nn.Linear(latent_dim, latent_dim)

        self.o_linear2 = nn.Linear(latent_dim, latent_dim)
        self.gated_linear2 = nn.Linear(latent_dim, latent_dim)

        self.norm6 = nn.LayerNorm(latent_dim)
        self.outnorm2 = nn.LayerNorm(latent_dim)

        self.ffn2 = FeedForward(d_dim=latent_dim, ff_dim=latent_dim,
                                output_dim=latent_dim, dropout=dropout,
                                ff_act_fn=ff_act_fn)

        self.dropout = nn.Dropout(dropout)

        # ----- KEGG projection -----
        self.kegg_norm = nn.LayerNorm(kegg_dim)
        self.kegg_proj = nn.Linear(kegg_dim, latent_dim)

        # Classifier: prot_rep + drug_rep + kegg_rep + init_drug_rep
        ffc_dim = latent_dim * 4
        self.cls_norm = nn.LayerNorm(ffc_dim)
        self.ffc = FeedForward(d_dim=ffc_dim, ff_dim=ffc_hidden_dims,
                               output_dim=1, dropout=dropout,
                               ff_act_fn=ff_act_fn)

    def forward(self, protein, drug, kegg, prot_mask=None, drug_mask=None):
        """
        protein  : [batch, N_prot, protein_dim]
        drug     : [batch, N_drug, drug_dim]
        kegg     : [batch, kegg_dim]
        prot_mask: [batch, N_prot]  (1=valid, 0=pad)
        drug_mask: [batch, N_drug]  (1=valid, 0=pad)

        Returns: logit, cross1_attn_weights, cross2_attn_weights,
                 prot_rep, drug_rep, combined_rep
        """
        projected_prot = self.prot_proj(protein)
        projected_drug = self.drug_proj(drug)

        batch_size, prot_seq_len, _ = projected_prot.size()
        _, drug_seq_len, _ = projected_drug.size()

        # ===== Cross1: Protein (Q) <- Drug (K, V) =====
        prot_q = self.prot_q(self.norm1(projected_prot))
        drug_k = self.drug_k(self.norm2(projected_drug))
        drug_v = self.drug_v(self.norm2(projected_drug))

        prot_q_re = prot_q.reshape(batch_size, prot_seq_len,
                                    self.cross1_mha_num_heads, self.cross1_head_dim).permute(0, 2, 1, 3)
        drug_k_re = drug_k.reshape(batch_size, drug_seq_len,
                                    self.cross1_mha_num_heads, self.cross1_head_dim).permute(0, 2, 1, 3)
        drug_v_re = drug_v.reshape(batch_size, drug_seq_len,
                                    self.cross1_mha_num_heads, self.cross1_head_dim).permute(0, 2, 1, 3)

        cross1_scores = torch.matmul(prot_q_re, drug_k_re.transpose(-2, -1)) / (self.cross1_head_dim ** 0.5)

        if drug_mask is not None:
            cross1_scores = cross1_scores.masked_fill(
                drug_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        cross1_attn_weights = F.softmax(cross1_scores, dim=-1)

        cross1_context = torch.matmul(cross1_attn_weights, drug_v_re)
        cross1_concat = cross1_context.permute(0, 2, 1, 3).reshape(
            batch_size, prot_seq_len, self.latent_dim)

        cross1_gated = cross1_concat * torch.sigmoid(self.gated_linear1(cross1_concat))
        cross1_lin = self.dropout(self.o_linear1(cross1_gated))

        cross1_att = projected_prot + cross1_lin
        cross1_prot_vec = self.outnorm1(
            projected_prot + self.dropout(self.ffn1(self.norm3(cross1_att)))
        )

        # ===== Cross2: Drug (Q) <- Protein (K, V) =====
        drug_q = self.drug_q(self.norm4(projected_drug))
        prot_k = self.prot_k(self.norm5(projected_prot))
        prot_v = self.prot_v(self.norm5(projected_prot))

        drug_q_re = drug_q.reshape(batch_size, drug_seq_len,
                                    self.cross2_mha_num_heads, self.cross2_head_dim).permute(0, 2, 1, 3)
        prot_k_re = prot_k.reshape(batch_size, prot_seq_len,
                                    self.cross2_mha_num_heads, self.cross2_head_dim).permute(0, 2, 1, 3)
        prot_v_re = prot_v.reshape(batch_size, prot_seq_len,
                                    self.cross2_mha_num_heads, self.cross2_head_dim).permute(0, 2, 1, 3)

        cross2_scores = torch.matmul(drug_q_re, prot_k_re.transpose(-2, -1)) / (self.cross2_head_dim ** 0.5)

        if prot_mask is not None:
            cross2_scores = cross2_scores.masked_fill(
                prot_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        cross2_attn_weights = F.softmax(cross2_scores, dim=-1)

        cross2_context = torch.matmul(cross2_attn_weights, prot_v_re)
        cross2_concat = cross2_context.permute(0, 2, 1, 3).reshape(
            batch_size, drug_seq_len, self.latent_dim)

        cross2_gated = cross2_concat * torch.sigmoid(self.gated_linear2(cross2_concat))
        cross2_lin = self.dropout(self.o_linear2(cross2_gated))

        cross2_att = projected_drug + cross2_lin
        cross2_drug_vec = self.outnorm2(
            projected_drug + self.dropout(self.ffn2(self.norm6(cross2_att)))
        )

        # ===== Pooling =====
        # prot_rep: mean pool over cross1 output (protein attended drug, cross-attn + FFN)
        if prot_mask is not None:
            p_bool = (prot_mask.unsqueeze(-1) == 0)
            masked_prot = cross1_prot_vec.masked_fill(p_bool, 0.0)
            valid_prot_len = prot_mask.sum(dim=1, keepdim=True).clamp(min=1)
            prot_rep = masked_prot.sum(dim=1) / valid_prot_len
        else:
            prot_rep = cross1_prot_vec.mean(dim=1)

        # drug_rep: mean pool over cross2 output (drug attended protein, cross-attn + FFN)
        if drug_mask is not None:
            d_bool = (drug_mask.unsqueeze(-1) == 0)
            masked_drug = cross2_drug_vec.masked_fill(d_bool, 0.0)
            valid_drug_len = drug_mask.sum(dim=1, keepdim=True).clamp(min=1)
            drug_rep = masked_drug.sum(dim=1) / valid_drug_len
        else:
            drug_rep = cross2_drug_vec.mean(dim=1)

        # Initial drug representation (before cross attention)
        if drug_mask is not None:
            d_mask_bool = (drug_mask.unsqueeze(-1) == 0)
            masked_drug = projected_drug.masked_fill(d_mask_bool, 0.0)
            valid_lengths = drug_mask.sum(dim=1, keepdim=True).clamp(min=1)
            init_drug_rep = masked_drug.sum(dim=1) / valid_lengths
        else:
            init_drug_rep = projected_drug.mean(dim=1)

        # KEGG projection
        kegg_rep = self.kegg_proj(self.kegg_norm(kegg))

        # Classifier: prot_rep + drug_rep + kegg_rep + init_drug_rep
        combined_rep = self.cls_norm(
            torch.cat([prot_rep, drug_rep, kegg_rep, init_drug_rep], dim=1)
        )
        logit = self.ffc(combined_rep)

        return logit, cross1_attn_weights, cross2_attn_weights, prot_rep, drug_rep, combined_rep


# ---------------------------------------------------------------------------
# Dataset (with KEGG)
# ---------------------------------------------------------------------------

class microbe_drug_dataset(Dataset):
    def __init__(self, indices, protein_cache, drug_tensor_dict, kegg_data, labels):
        self.indices = indices
        self.protein_cache = protein_cache
        self.drug_tensor_dict = drug_tensor_dict
        self.kegg_data = kegg_data
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_row = self.indices[index]
        microbe_name = data_row['bacterial strain']
        drug_name = data_row['DrugName']

        prot_tensor = self.protein_cache[microbe_name]
        drug_tensor = self.drug_tensor_dict[drug_name]
        kegg_tensor = self.kegg_data[microbe_name]
        label = self.labels[index]

        return prot_tensor, drug_tensor, kegg_tensor, label


def collate_fn(batch):
    proteins, drugs, keggs, labels = zip(*batch)

    prot_lengths = [len(p) for p in proteins]
    padded_proteins = pad_sequence(proteins, batch_first=True, padding_value=0)
    prot_mask = torch.zeros(len(proteins), padded_proteins.size(1))
    for i, length in enumerate(prot_lengths):
        prot_mask[i, :length] = 1

    drug_lengths = [len(d) for d in drugs]
    padded_drugs = pad_sequence(drugs, batch_first=True, padding_value=0)
    drug_mask = torch.zeros(len(drugs), padded_drugs.size(1))
    for i, length in enumerate(drug_lengths):
        drug_mask[i, :length] = 1

    keggs = torch.stack(keggs)
    labels = torch.stack(labels)

    return padded_proteins, padded_drugs, keggs, labels, prot_mask, drug_mask


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def make_class_balanced_sampler(y_tensor):
    y_np = y_tensor.detach().cpu().numpy().flatten()
    class_counts = np.bincount(y_np.astype(int))
    if len(class_counts) < 2:
        return None
    class_weights = 1. / (class_counts + 1e-6)
    sample_weights = class_weights[y_np.astype(int)]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
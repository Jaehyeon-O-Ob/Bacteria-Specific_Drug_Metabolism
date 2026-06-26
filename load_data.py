import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')
data_dir = os.path.join(current_dir, 'input')

def load_protein_data(data_dir: str, bac_names) -> dict:
    protein_data = {}
    for bac in bac_names:
        pt_path = os.path.join(data_dir, f"{bac}_esmc_600m.pt")
        try:
            tensor = torch.load(pt_path, map_location=torch.device('cpu'), weights_only=True)
        except TypeError:
            tensor = torch.load(pt_path, map_location=torch.device('cpu'))
        if isinstance(tensor, dict):
            tensor = torch.stack(list(tensor.values()))
        if tensor.dim() == 3: tensor = tensor.squeeze(0)
        if tensor.dim() == 1: tensor = tensor.unsqueeze(0)
        protein_data[bac] = tensor.float()
    return protein_data

def load_drug_data(data_dir: str) -> dict:
    drug_sources = {'molE': 'drug_node_representations.pt'}
    drug_data = {}
    for rep_name, filename in drug_sources.items():
        pt_path = os.path.join(data_dir, filename)
        try:
            raw = torch.load(pt_path, map_location="cpu", weights_only=True)
        except TypeError:
            raw = torch.load(pt_path, map_location="cpu")
        rep_dict = {}
        for drug_name, embedding in raw.items():
            if isinstance(embedding, list): embedding = torch.stack(embedding)
            if embedding.dim() == 3: embedding = embedding.squeeze(0)
            rep_dict[drug_name] = embedding
        drug_data[rep_name] = rep_dict
    return drug_data

def load_kegg_data(kegg_df) -> dict:
    df = kegg_df.set_index('bacterial strain')
    kegg_data = {}
    for strain, row in df.iterrows():
        kegg_data[strain] = torch.tensor(row.values, dtype=torch.float32)
    return kegg_data
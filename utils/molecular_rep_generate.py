from rdkit import Chem
import os

import pandas as pd
import os
import torch
import sys
import yaml
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from mole_antimicrobial_potential.workflow.models.ginet_concat import GINet
from mole_antimicrobial_potential.workflow.dataset.dataset_representation import MoleculeDataset

def extract_node_sequence(graph):
    """
    input = graph data

    single graph objective [number of atom, feature dim] > molecular representation

    concat all of 5 layers = feature dim = 1000
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = os.path.join(current_dir, 'mole_antimicrobial_potential', 'pretrained_model', 'model_ginconcat_btwin_100k_d8000_l0.0001')

    with open(os.path.join(model_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    model = GINet(**config["model"]).to(device)
    state_dict = torch.load(os.path.join(model_dir, 'model.pth'), map_location=device)
    model.load_my_state_dict(state_dict=state_dict)
    model.eval()

    with torch.no_grad():
        graph = graph.to(device)
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        h = model.x_embedding1(x[:,0]) + model.x_embedding2(x[:,1])
        
        h_list = []
        for i in range(model.num_layer):
            h = model.gnns[i](h, edge_index, edge_attr)
            h = model.batch_norms[i](h)
            
            if i < model.num_layer - 1:
                h = F.relu(h)
                
            h_list.append(h)
            
        node_representation = torch.cat(h_list, dim=1)
        
    return node_representation.cpu()
import argparse
import os
import copy
import pandas as pd
import numpy as np
import torch
import json

from torch.utils.data import DataLoader

from model.multi_modal_bi_mha_model import multi_modal_bi_cross_mha, collate_fn, microbe_drug_dataset
from load_data import load_kegg_data

# Make sure the external dependencies (ESM-C, MolE, KofamScan + KOfam database)
# are downloaded before importing the modules that rely on them (same as
# prediction.py), so this script also works on a fresh installation.
from utils.setup_dependencies import ensure_all_dependencies
ensure_all_dependencies()

from utils.bacteria_rep_generate import ko_id_mapping, ko_filtering, extract_protein_sequence, kegg_scored_rep
from utils.molecular_rep_generate import extract_node_sequence
from utils.mole_antimicrobial_potential.workflow.dataset.dataset_representation import MoleculeDataset
from utils.input_utils import parse_input_txt, collect_user_proteins

def main():
    parser = argparse.ArgumentParser(description="Extract combined representations from the trained multi-modal models.")

    parser.add_argument(
            '--input_txt',
            type=str,
            required=True,
            help="the text file for prediction (ex: input.txt)"
    )
    
    parser.add_argument(
            '--output_csv',
            type=str,
            required=False,
            default='combined_representations.csv',
            help="Base filename for the extracted representations. One file per fold-specific model is "
                 "saved, e.g. combined_representations_fold_1.csv ... _fold_5.csv "
                 "(Default: combined_representations.csv)"
    )

    parser.add_argument(
            '--target_protein',
            type=str,
            required=False,
            default=None,
            help="User-defined protein ID(s), comma-separated, added to the model input of every bacterial "
                 "strain (same as prediction.py). A per-record 'target protein:' line in the input text "
                 "file can be used instead or in addition (Default: None)"
    )

    parser.add_argument(
            '--include_proteins',
            type=str,
            required=False,
            default=None,
            help="Comma-separated protein IDs to force-include in the model input (knowledge-based "
                 "augmentation), same as prediction.py (Default: None)"
    )

    parser.add_argument(
            '--esm_max_tokens',
            type=int,
            required=False,
            default=None,
            help="Max tokens per batch for ESM-C model (Default: 2048)"
    )

    parser.add_argument(
            '--cpu_cores',
            type=int,
            required=False,
            default=None,
            help="Number of CPU cores for kofam_scan (Default: 8)"
    )
        
    args = parser.parse_args()

    if args.esm_max_tokens is None:
        print("[DEBUG] --esm_max_tokens is not set. Using default value: 2048")
        args.esm_max_tokens = 2048
    else:
        print(f"[DEBUG] Using specified --esm_max_tokens: {args.esm_max_tokens}")

    if args.cpu_cores is None:
        print("[DEBUG] --cpu_cores is not set. Using default value: 8")
        args.cpu_cores = 8
    else:
        print(f"[DEBUG] Using specified --cpu_cores: {args.cpu_cores}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'model')
    data_dir = os.path.join(current_dir, 'input')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data load
    hp_config_path = os.path.join(model_dir, 'model_config.json')

    with open(hp_config_path, 'r') as f:
        hp_config = json.load(f)['hyperparams']

    pred_df = parse_input_txt(args.input_txt)
    
    # Check required columns
    req_cols = ['drug', 'drug smiles code', 'bacterial strain name']
    for col in req_cols:
        if col not in pred_df.columns:
            raise ValueError(f"Missing required key: '{col}' in input text file.")

    # Convert to standard format
    pred_df = pred_df.rename(columns={
        'drug': 'DrugName',
        'bacterial strain name': 'bacterial strain'
    })

    bacteria_list = pred_df['bacterial strain'].unique().tolist()
    
    # Check if fasta files exist
    for bac in bacteria_list:
        fasta_path = os.path.join(data_dir, f"{bac}_proteins.fasta")
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"Protein sequence fasta file '{bac}_proteins.fasta' not found in input folder.")

    # 1. produce representation - Bacteria
    print("Generating Bacteria Representations...")
    ko_id_mapping(bacteria_list, cpu_cores=args.cpu_cores)
    # User-defined proteins are added exactly as in prediction.py, so that the
    # extracted representations correspond to the same model input.
    include_dict = collect_user_proteins(pred_df, 'bacterial strain',
                                         target_protein=args.target_protein,
                                         include_proteins=args.include_proteins)
    for bac, prots in include_dict.items():
        if prots:
            print(f"User-defined proteins for {bac}: {', '.join(prots)}")
    ko_filtering(bacteria_list, include_dict)
    extract_protein_sequence(bacteria_list, MAX_TOKENS_PER_BATCH=args.esm_max_tokens)
    kegg_df = kegg_scored_rep(bacteria_list)

    # 2. produce representation - Drug
    print("Generating Drug Representations...")
    smile_col = 'drug smiles code'
    id_col = 'DrugName'
    mol_dataset = MoleculeDataset(pred_df, smile_col, id_col)
    
    drug_tensor_dict = {}
    for i in range(len(mol_dataset)):
        graph = mol_dataset[i]
        drug_id = graph.chem_id
        if drug_id not in drug_tensor_dict:
            rep = extract_node_sequence(graph)
            drug_tensor_dict[drug_id] = rep

    # 3. load all data representation
    print("Loading all representations...")
    protein_data = {}
    for bac in bacteria_list:
        pt_path = os.path.join(data_dir, f"{bac}_esmc_600m.pt")
        tensor_dict = torch.load(pt_path, map_location='cpu')
        
        stacked_tensor = torch.stack(list(tensor_dict.values())).float()
        
        if stacked_tensor.dim() == 3: stacked_tensor = stacked_tensor.squeeze(0)
        if stacked_tensor.dim() == 1: stacked_tensor = stacked_tensor.unsqueeze(0)
        protein_data[bac] = stacked_tensor

    kegg_data = load_kegg_data(kegg_df)
    
    # Dummy labels for DataLoader
    labels = torch.zeros(len(pred_df), dtype=torch.float)

    dataset = microbe_drug_dataset(
        pred_df.to_dict('records'),
        protein_data,
        drug_tensor_dict,
        kegg_data,
        labels
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 4. Extract Representations
    print("Extracting Combined Representations...")
    protein_dim = 1152
    drug_dim = 1000
    kegg_dim = 204
    latent_dim = hp_config['latent_dim']
    cross1_heads = hp_config['cross1_mha_num_heads']
    cross2_heads = hp_config['cross2_mha_num_heads']
    ffc_dims = hp_config['ffc_hidden_dims']
    dropout = hp_config['dropout']
    activation = hp_config.get('activation', 'relu')

    models = []
    for i in range(1, 6):
        m = multi_modal_bi_cross_mha(
            protein_dim=protein_dim,
            drug_dim=drug_dim,
            kegg_dim=kegg_dim,
            latent_dim=latent_dim,
            cross1_mha_num_heads=cross1_heads,
            cross2_mha_num_heads=cross2_heads,
            ffc_hidden_dims=ffc_dims,
            ff_act_fn=activation,
            dropout=dropout
        ).to(device)
        m.load_state_dict(torch.load(os.path.join(model_dir, f'multi modal bi mha {i}.pth'), map_location=device))
        m.eval()
        models.append(m)

    # The five fold-specific models were trained independently, so their latent
    # coordinates are not aligned with each other (the same dimension index does
    # not encode the same feature across models). The representations are
    # therefore NOT averaged across models; the representation of each
    # fold-specific model is saved to a separate file.
    fold_representations = {i: [] for i in range(1, len(models) + 1)}

    for batch_idx, batch in enumerate(dataloader):
        padded_proteins, padded_drugs, keggs, _, prot_mask, drug_mask = [b.to(device) if torch.is_tensor(b) else b for b in batch]

        with torch.no_grad():
            for fold, m in enumerate(models, start=1):
                _, _, _, _, _, combined_rep = m(
                    padded_proteins, padded_drugs, keggs, prot_mask, drug_mask
                )
                fold_representations[fold].append(combined_rep.cpu().numpy()[0])  # Shape: [rep_dim]

    # 5. Save Results
    # Restore original column names
    pred_df = pred_df.rename(columns={
        'DrugName': 'drug',
        'bacterial strain': 'bacterial strain name'
    })
    meta_df = pred_df[['drug', 'bacterial strain name', 'drug smiles code']].reset_index(drop=True)

    base, ext = os.path.splitext(args.output_csv)
    ext = ext or '.csv'
    for fold, reps in fold_representations.items():
        rep_dim = reps[0].shape[0]
        rep_columns = [f"dim_{i+1}" for i in range(rep_dim)]
        rep_df = pd.DataFrame(reps, columns=rep_columns)

        # Combine original metadata with the representations of this fold model
        final_df = pd.concat([meta_df, rep_df], axis=1)

        output_path = os.path.join(current_dir, f"{base}_fold_{fold}{ext}")
        final_df.to_csv(output_path, index=False)
        print(f"Fold {fold} combined representations saved to {output_path}")

    print("Extraction complete.")

if __name__ == "__main__":
    main()

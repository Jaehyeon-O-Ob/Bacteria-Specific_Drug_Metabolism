import xgboost as xgb
import argparse
import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

from torch.utils.data import DataLoader
import sys
import importlib

from model.multi_modal_bi_mha_model import multi_modal_bi_cross_mha, collate_fn, microbe_drug_dataset
from load_data import load_kegg_data

# Make sure the external dependencies (ESM-C, MolE, KofamScan + KOfam database)
# are downloaded before importing the modules that rely on them. The first run
# can take a while because of the large model/database downloads.
from utils.setup_dependencies import ensure_all_dependencies
ensure_all_dependencies()

from utils.bacteria_rep_generate import ko_id_mapping, ko_filtering, extract_protein_sequence, kegg_scored_rep
from utils.molecular_rep_generate import extract_node_sequence
from utils.mole_antimicrobial_potential.workflow.dataset.dataset_representation import MoleculeDataset
from utils.ec_analysis import run_ec_attention_analysis
from utils.input_utils import parse_input_txt, collect_user_proteins

def main():
    parser = argparse.ArgumentParser(description="Run prediction with a specific Text file and Target Protein.")

    parser.add_argument(
            '--input_txt',
            type=str,
            required=True,
            help="the text file for prediction (ex: input.txt)"
    )

    parser.add_argument(
            '--target_protein',
            type=str,
            required=False,
            default=None,
            help="User-defined protein ID(s), comma-separated, applied to every bacterial strain in the "
                 "input file. They are automatically included in the prediction input and their EC numbers "
                 "and atom-attention maps are reported. A per-record 'target protein:' line in the input "
                 "text file can be used instead or in addition (Default: None)"
    )

    parser.add_argument(
            '--include_proteins',
            type=str,
            required=False,
            default=None,
            help="Comma-separated protein IDs to force-include in the prediction input "
                 "(knowledge-based augmentation), added on top of the automatic KO filtering. "
                 "Their EC numbers and atom-attention maps are reported as user-defined proteins "
                 "(e.g., 'WP_011107642.1,WP_008760980.1')"
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
            help="Number of CPU cores for kofam_scan and xgboost (Default: 8)"
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
    attn_dir = os.path.join(current_dir, 'attention analysis')

    if not os.path.exists(attn_dir):
        os.makedirs(attn_dir)

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
    
    # User-defined proteins force-included in the prediction input, added on top
    # of the automatic KO-based filtering (built once per bacterial strain):
    #   - --target_protein / --include_proteins (applied to every strain), and
    #   - the optional 'target protein:' line of each record in the input text
    #     file (applied to the strain of that record).
    # Their EC numbers and atom-attention maps are reported in the analysis.
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
    protein_keys_dict = {}
    for bac in bacteria_list:
        pt_path = os.path.join(data_dir, f"{bac}_esmc_600m.pt")
        tensor_dict = torch.load(pt_path, map_location='cpu')
        
        protein_keys_dict[bac] = list(tensor_dict.keys())
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

    # 4. prediction
    print("Running Predictions...")
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
    xgbs = []
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
        
        x = xgb.XGBClassifier(n_jobs=args.cpu_cores)
        x.load_model(os.path.join(model_dir, f'xgb {i}.json'))
        xgbs.append(x)

    all_preds = []
    all_probs = []
    all_rep_ecs = []

    def _safe_name(name):
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name)).strip("_")

    for batch_idx, batch in enumerate(dataloader):
        padded_proteins, padded_drugs, keggs, _, prot_mask, drug_mask = [b.to(device) if torch.is_tensor(b) else b for b in batch]

        fold_probs = []
        att1_list = []
        att2_list = []

        # Fold-wise ensemble (as in the paper): each fold-specific Bi-MHA model
        # produces its own combined representation, which is classified by the
        # XGBoost model of the same fold; the five probabilities are averaged.
        with torch.no_grad():
            for m, x in zip(models, xgbs):
                logit, cr1_att, cr2_att, prot_rep, drug_rep, combined_rep = m(
                    padded_proteins, padded_drugs, keggs, prot_mask, drug_mask
                )
                fold_probs.append(x.predict_proba(combined_rep.cpu().numpy())[:, 1][0])
                att1_list.append(cr1_att)
                att2_list.append(cr2_att)

        final_prob = float(np.mean(fold_probs))
        pred_label = 1 if final_prob >= 0.5 else 0
        all_preds.append(pred_label)
        all_probs.append(final_prob)

        # Attention-based interpretation (Case Study procedure of the paper):
        # protein ranking by mean fold rank -> EC sub-subclass enrichment of the
        # top 10% -> all EC classes with FE > 1 and P <= 0.05 are reported as
        # representative candidates with EC-averaged and per-protein atom maps
        # (+ user-defined proteins)
        data_row = pred_df.iloc[batch_idx]
        bac = data_row['bacterial strain']
        drug_name = data_row['DrugName']
        drug_smiles = data_row['drug smiles code']

        result = run_ec_attention_analysis(
            drug_name=drug_name,
            drug_smiles=drug_smiles,
            bacteria_name=bac,
            protein_ids=protein_keys_dict[bac],
            cross1_list=att1_list,
            cross2_list=att2_list,
            kofam_tsv=os.path.join(data_dir, f"{bac}_kegg_result.tsv"),
            fasta_path=os.path.join(data_dir, f"{bac}_proteins.fasta"),
            save_dir=os.path.join(attn_dir, _safe_name(f"{drug_name}_{bac}")),
            user_proteins=include_dict.get(bac, []),
        )
        summary = result['enriched_summary']
        all_rep_ecs.append('-' if summary.empty else '; '.join(summary['EC_level3']))

    pred_df['Prediction'] = all_preds
    pred_df['Probability'] = all_probs
    pred_df['Representative EC (top 10%, P<=0.05)'] = all_rep_ecs
    
    # Restore original column names
    pred_df = pred_df.rename(columns={
        'DrugName': 'drug',
        'bacterial strain': 'bacterial strain name'
    })
    
    output_path = os.path.join(current_dir, 'prediction_results.csv')
    pred_df.to_csv(output_path, index=False)
    print(f"Prediction complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
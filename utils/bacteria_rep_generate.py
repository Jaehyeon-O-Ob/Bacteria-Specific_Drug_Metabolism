import os
import sys
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
_old_request = requests.Session.request
def _new_request(*args, **kwargs):
    kwargs['verify'] = False
    return _old_request(*args, **kwargs)
requests.Session.request = _new_request
import subprocess
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
esm_path = os.path.join(current_dir, 'esm_official')

parent_dir = os.path.dirname(current_dir)
bac_dir = os.path.join(parent_dir, 'input')
kofam_db_dir = os.path.join(current_dir, 'kofam_db')    # update 260531

if esm_path not in sys.path:
    sys.path.insert(0, esm_path)

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def ko_id_mapping(bacteria_list: list,
                  cpu_cores:int):
    """
    bacteria_list for bacteria name
    cpu cores: cpu number for usage of kofamscan
    if the user have target proteins to use or analyze the metabolism
    """
    for bac in bacteria_list:
        file_name = os.path.join(bac_dir, f"{bac}_proteins.fasta")
        output_path = os.path.join(bac_dir, f"{bac}_kegg_result.tsv")

        if os.path.exists(output_path):
            print(f"KofamScan result already exists for {bac}. Skipping annotation.")
            continue

        exec_annotation_path = os.path.join(kofam_db_dir, "kofam_scan-1.3.0", "exec_annotation")

        command = [
            exec_annotation_path,
            "-f", "mapper",
            "-o", output_path,
            file_name,
            "-p", os.path.join(kofam_db_dir, 'profiles'),
            "-k", os.path.join(kofam_db_dir, 'ko_list'),
            "--cpu", str(cpu_cores)
        ]

        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"{bac}: Success"
        except Exception as e:
            return f"{bac}: Error ({e})"

def ko_filtering(bacteria_list:list,
                 include_proteins:dict = None):

    """
    Build the filtered protein set used for prediction.

    For each bacterium the set is the union of:
      1. proteins annotated with a KO listed in `valid metabolism kos`
         (automatic KofamScan-based filtering), and
      2. proteins supplied via `include_proteins` (knowledge-based augmentation),
         which are added even if they were dropped by the automatic filter.

    include_proteins is a dict like {"bacteria name": ["protein id", ...]}.
    The protein ids must exist in the bacterium's original protein fasta;
    ids that are not found are reported and ignored.
    """

    valid_metabolism_kos = set()
    kos_path = os.path.join(current_dir, "valid metabolism kos (260513).txt")
    with open(kos_path, "r") as f:
        for line in f:
            line = line.replace('\n', '')
            valid_metabolism_kos.add(line)

    for bac in bacteria_list:
        mapped_ids = set()
        unmapped_ids = set()

        kofam_res = pd.read_csv(os.path.join(bac_dir, f"{bac}_kegg_result.tsv"), sep='\t', header=None, names=['protein_id', 'ko_id'])

        for _, row in kofam_res.iterrows():
            p_id = str(row['protein_id']).strip()
            k_id = str(row['ko_id']).strip()

            if k_id == 'nan' or k_id == '':
                unmapped_ids.add(p_id)
            elif k_id in valid_metabolism_kos:
                mapped_ids.add(p_id)
            else:
                pass

        auto_count = len(mapped_ids)

        all_prots = os.path.join(bac_dir, f"{bac}_proteins.fasta")
        mapped_prots = os.path.join(bac_dir, f"{bac}_filtered_prots.fasta")
        available_ids = {record.id for record in SeqIO.parse(all_prots, "fasta")}

        # Knowledge-based augmentation: force-include requested proteins that
        # exist in the original fasta but were dropped by the automatic filter.
        requested = include_proteins.get(bac, []) if include_proteins else []
        injected_ids = []
        missing_ids = []
        for pid in requested:
            if pid in available_ids:
                if pid not in mapped_ids:
                    injected_ids.append(pid)
                mapped_ids.add(pid)
            else:
                missing_ids.append(pid)

        with open(mapped_prots, "w") as m_out:
            m_counts = 0
            total_len = 0

            for record in SeqIO.parse(all_prots, "fasta"):
                total_len += 1
                if record.id in mapped_ids:
                    SeqIO.write(record, m_out, 'fasta')
                    m_counts += 1

        print(f"{bac} has {m_counts} proteins filtered from {total_len} proteins "
              f"({auto_count} by automatic KO filtering, {len(injected_ids)} added by knowledge).")
        if injected_ids:
            print(f"  Knowledge-injected proteins for {bac}: {', '.join(injected_ids)}")
        if missing_ids:
            print(f"  [WARNING] Requested proteins not found in {bac} fasta (ignored): {', '.join(missing_ids)}")


def extract_protein_sequence(bacteria_list: list, 
                             model_name: str = "esmc_600m", 
                             MAX_TOKENS_PER_BATCH=2048):
    """
    Extract the protein sequence for the given sequence
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ESMC.from_pretrained(model_name).to(device).eval()

    print(f"ESM C {model_name} loaded!")

    MAX_TOKENS_PER_BATCH = MAX_TOKENS_PER_BATCH # Safe for 16GB VRAM with Flash Attention (e.g. 4x2048, 16x512)
    MAX_SEQ_LEN = 2046  # ESM-C context window = 2048 total tokens (CLS + residues + EOS)
                        # Max residues = 2048 - 2 = 2046

    for bac in bacteria_list:
        print(f"Processing {bac}.....")

        prot_path = os.path.join(bac_dir, f"{bac}_filtered_prots.fasta")

        if not os.path.exists(prot_path):
            print(f"There is no protein sequence file for {bac}")
            continue

        sequences = list(SeqIO.parse(prot_path, "fasta"))

        sequences.sort(key=lambda x: len(x.seq))

        batches = []
        current_batch = []
        current_max_len = 0

        for rec in sequences:
            seq_len = min(MAX_SEQ_LEN, len(rec.seq))
            projected_max = max(current_max_len, seq_len + 2) # +2 for CLS, EOS

            if len(current_batch) > 0 and (len(current_batch) + 1) * projected_max > MAX_TOKENS_PER_BATCH:
                batches.append(current_batch)
                current_batch = [rec]
                current_max_len = seq_len + 2
            else:
                current_batch.append(rec)
                current_max_len = projected_max
                
        if current_batch:
            batches.append(current_batch)

        embedding_results = {}
        processed_count = 0
        with torch.no_grad():
            for batch in batches:

                # Extract truncated sequences
                seq_strs = [str(rec.seq)[:MAX_SEQ_LEN] for rec in batch]

                # True batch processing on GPU
                input_ids = model._tokenize(seq_strs).to(device)
                outputs = model(input_ids)

                # Retrieve mean-pooled embeddings for each sequence in the batch
                for idx, rec in enumerate(batch):
                    truncate_len = len(seq_strs[idx])
                    # Mean pooling over residue tokens only (CLS at 0 and EOS at -1 excluded)
                    # Consistent with ESM-2 official extract.py (Meta, line 118)
                    mean_emb = outputs.embeddings[idx, 1 : truncate_len + 1].mean(dim=0).detach().cpu()
                    embedding_results[rec.id] = mean_emb

                if device == 'cuda':
                    torch.cuda.empty_cache()
                
                processed_count += len(batch)
                if processed_count % 300 < len(batch) or processed_count == len(sequences):
                    print(f"  [{bac}] Processed {processed_count}/{len(sequences)} proteins")

        prot_emb = embedding_results

        output_path = os.path.join(bac_dir, f"{bac}_{model_name}.pt")
        torch.save(prot_emb, output_path)
        print(f"Saved {len(prot_emb)} embeddings.")

def kegg_scored_rep(bacteria_list: list):
    """
    Here is one of the modalities to train to main model, which is kegg pathway-based score vector
    """
    ko_pathway_mapping = pd.read_csv(os.path.join(current_dir, 'ko_pathway_mapping.csv'))
    pathway_info = pd.read_csv(os.path.join(current_dir, 'all_bacteria_pathways.csv'))

    pathway_info['Map ID'] = 'map' + pathway_info['Map ID'].astype(str).str.zfill(5)
    id_to_name = dict(zip(pathway_info['Map ID'], pathway_info['Pathway Name']))

    ko_counts = ko_pathway_mapping.groupby('ko_id')['pathway_id'].nunique().to_dict()
    target_pathways = ko_pathway_mapping['pathway_id'].unique().tolist()
    ko_to_pathways = ko_pathway_mapping.groupby('ko_id')['pathway_id'].apply(list).to_dict()

    all_bac_kegg_scores = []

    for bac in bacteria_list:
        kegg_path = os.path.join(bac_dir, f"{bac}_kegg_result.tsv")
        kegg_df = pd.read_csv(kegg_path, sep='\t', names=['prot_id', 'ko_id'])

        bac_ko_set = set(kegg_df['ko_id'].unique())

        pathway_scores = {path: 0.0 for path in target_pathways}
        pathway_scores['bacterial strain'] = bac

        for ko in bac_ko_set:
            if ko in ko_to_pathways:
                p_list = ko_to_pathways[ko]
                
                all_prots = ko_counts[ko]

                weight = 1.0 / all_prots

                for path_id in p_list:
                    if path_id in pathway_scores:
                        pathway_scores[path_id] += weight

        all_bac_kegg_scores.append(pathway_scores)

    bac_kegg_rep_df = pd.DataFrame(all_bac_kegg_scores)

    cols = ['bacterial strain'] + target_pathways
    bac_kegg_rep_df = bac_kegg_rep_df[cols]
    bac_kegg_rep_df = bac_kegg_rep_df.rename(columns=id_to_name)

    return bac_kegg_rep_df
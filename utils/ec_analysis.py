"""
ec_analysis.py
==============

Attention-based interpretation following the Case Study procedure of the paper
("Prediction of bacteria-specific drug metabolism using a cross-attention-based
model", Briefings in Bioinformatics 2026, bbag578; Supplementary Methods, Case
Study section).

Procedure
---------
1. Protein-level attention (Cross 2; drug atom tokens = queries, bacterial
   protein tokens = keys/values) is averaged over heads and drug atoms for each
   of the five fold-specific models. Proteins are ranked within each fold
   (rank 1 = highest attention) and ordered by their mean rank across the five
   folds (smaller mean rank = higher priority).
2. The top 10% of the input protein pool (rounded up) is selected. The complete
   filtered input protein pool (the proteins actually given to the model) is
   used as the enrichment background.
3. Each protein is assigned the EC numbers of its first KofamScan KO hit using
   the KEGG ``link/enzyme/ko`` table shipped in
   ``utils/kegg_ko_to_ec (261105).tsv``. Enrichment is tested at the
   three-component EC sub-subclass level (e.g. EC 3.1.1) with a one-sided
   Fisher's exact test. Fold enrichment = (a/k)/(M/N). Every category with
   fold enrichment > 1 and unadjusted p-value <= 0.05 is reported as a
   representative EC candidate, so that the user can inspect and select among
   them (``EC_candidates_top10pct.csv`` / ``EC_candidate_proteins_top10pct.csv``).
4. Atom-level attention (Cross 1; bacterial protein tokens = queries, drug atom
   tokens = keys/values) is averaged over heads and element-wise over the five
   folds. For each representative EC class, the rows of its member proteins
   inside the top 10% are averaged to obtain an EC-level atom-attention map; the
   map of each member protein (its own row) is also saved for inspection. For
   user-defined proteins, the protein's own row is used.
"""

import os
import re
import math

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

current_dir = os.path.dirname(os.path.abspath(__file__))
KO_EC_PATH = os.path.join(current_dir, "kegg_ko_to_ec (261105).tsv")

TOP_FRACTION = 0.10
P_THRESHOLD = 0.05
SIG_COL = "Representative(FE>1 & P<=0.05)"


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------
def load_ko_to_ec(path: str = KO_EC_PATH) -> dict:
    """KEGG link/enzyme/ko table ('ko:K00001<TAB>ec:1.1.1.1') -> {KO: [EC, ...]}"""
    ko_to_ec = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            ko = parts[0].replace("ko:", "").strip()
            ec = parts[1].replace("ec:", "").strip()
            ko_to_ec.setdefault(ko, []).append(ec)
    return ko_to_ec


def load_first_ko(kofam_tsv: str) -> dict:
    """First KO assigned to each protein in a KofamScan 'mapper' output."""
    first_ko = {}
    with open(kofam_tsv, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pid, ko = parts[0].strip(), parts[1].strip()
                if ko and pid not in first_ko:
                    first_ko[pid] = ko
    return first_ko


def load_fasta_descriptions(fasta_path: str) -> dict:
    """{protein id: header description} from a FASTA file."""
    desc = {}
    with open(fasta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split(None, 1)
                if parts:
                    desc[parts[0]] = parts[1] if len(parts) > 1 else ""
    return desc


def _level3(ec: str):
    parts = ec.split(".")
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    return None


# ---------------------------------------------------------------------------
# Attention aggregation
# ---------------------------------------------------------------------------
def aggregate_fold_attention(cross1_list, cross2_list):
    """
    cross1_list: per-fold Cross 1 attention, each [1, heads, N_prot, N_atom]
    cross2_list: per-fold Cross 2 attention, each [1, heads, N_atom, N_prot]

    Returns
    -------
    fold_prot_attn : [F, N_prot] protein attention per fold (heads & atoms averaged)
    fold_prot_rank : [F, N_prot] 1-indexed descending rank per fold
    ensemble_cross1: [N_prot, N_atom] fold-averaged Cross 1 attention (heads averaged)
    """
    fold_prot_attn, fold_prot_rank, fold_c1 = [], [], []
    for c1, c2 in zip(cross1_list, cross2_list):
        c1 = c1.detach().float().cpu()
        c2 = c2.detach().float().cpu()
        fold_c1.append(c1.squeeze(0).mean(dim=0).numpy())
        prot_attn = c2.squeeze(0).mean(dim=0).mean(dim=0).numpy()
        n = len(prot_attn)
        rank = np.empty(n, dtype=int)
        rank[np.argsort(-prot_attn, kind="stable")] = np.arange(1, n + 1)
        fold_prot_attn.append(prot_attn)
        fold_prot_rank.append(rank)
    return (np.array(fold_prot_attn), np.array(fold_prot_rank),
            np.mean(np.array(fold_c1), axis=0))


def build_protein_table(protein_ids, fold_prot_attn, fold_prot_rank,
                        first_ko, ko_to_ec, descriptions, user_proteins=None,
                        top_fraction=TOP_FRACTION):
    """One row per input protein, sorted by mean fold rank (ascending)."""
    user_proteins = set(user_proteins or [])
    n_folds = fold_prot_attn.shape[0]
    rows = []
    for idx, pid in enumerate(protein_ids):
        ko = first_ko.get(pid, "-")
        ecs = sorted(set(ko_to_ec.get(ko, [])))
        l3 = sorted({l for l in (_level3(e) for e in ecs) if l})
        row = {
            "protein_index": idx,
            "protein_id": pid,
            "description": descriptions.get(pid, ""),
            "user_defined": pid in user_proteins,
            "kegg_ko(first)": ko,
            "ec_numbers": ";".join(ecs) if ecs else "-",
            "ec_level3": ";".join(l3) if l3 else "-",
        }
        for f in range(n_folds):
            row[f"fold_{f+1}_attn"] = fold_prot_attn[f, idx]
        for f in range(n_folds):
            row[f"fold_{f+1}_rank"] = int(fold_prot_rank[f, idx])
        row["mean_attn"] = float(fold_prot_attn[:, idx].mean())
        row["mean_rank"] = float(fold_prot_rank[:, idx].mean())
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("mean_rank", kind="mergesort").reset_index(drop=True)
    df.insert(0, "order", np.arange(1, len(df) + 1))
    top_n = int(math.ceil(len(df) * top_fraction))
    df[f"in_top{int(round(top_fraction * 100))}pct"] = df["order"] <= top_n
    return df, top_n


def ec_level3_enrichment(prot_df, top_n, p_threshold=P_THRESHOLD):
    """One-sided Fisher's exact test for every EC sub-subclass in the pool."""
    total = len(prot_df)
    top_mask = prot_df["order"] <= top_n
    sets = prot_df["ec_level3"].apply(lambda s: set() if s == "-" else set(s.split(";")))
    terms = sorted(set().union(*sets.tolist())) if total else []

    rows = []
    for term in terms:
        member = sets.apply(lambda s: term in s)
        a = int((member & top_mask).sum())
        m = int(member.sum())
        b = m - a
        c = top_n - a
        d = (total - top_n) - b
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        fe = (a / top_n) / (m / total) if m > 0 and top_n > 0 else 0.0
        members_top = prot_df[member & top_mask]
        rows.append({
            "EC_level3": f"EC {term}",
            "N_background": total,
            "N_top": top_n,
            "N_class_in_background": m,
            "N_class_in_top": a,
            "Fold_enrichment": fe,
            "P_value(Fisher_one_sided)": p,
            "Top_protein_ids": "; ".join(members_top["protein_id"]),
            "Top_protein_descriptions": "; ".join(members_top["description"]),
            "Top_protein_full_EC": "; ".join(members_top["ec_numbers"]),
            "Top_protein_order": "; ".join(str(o) for o in members_top["order"]),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res.sort_values(["P_value(Fisher_one_sided)", "EC_level3"], kind="mergesort").reset_index(drop=True)
    # Benjamini-Hochberg q-values (reported for reference only; selection uses unadjusted p)
    mtests = len(res)
    q = res["P_value(Fisher_one_sided)"].values * mtests / (np.arange(mtests) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res["FDR_q_value(BH)"] = np.clip(q, None, 1.0)
    res["Representative(FE>1 & P<=0.05)"] = (res["Fold_enrichment"] > 1) & (res["P_value(Fisher_one_sided)"] <= p_threshold)
    return res


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _atom_labels(smiles, n_tokens):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, [f"A{i}" for i in range(n_tokens)]
    mol_h = Chem.AddHs(mol)
    labels = [f"{a.GetSymbol()}{a.GetIdx()}" for a in mol_h.GetAtoms()]
    if len(labels) < n_tokens:
        labels += [f"A{i}" for i in range(len(labels), n_tokens)]
    return mol, labels[:n_tokens]


def draw_atom_attention(smiles, atom_attn, save_path, cmap_name="Reds"):
    """2D atom-attention map (heavy atoms; darker red = higher relative attention)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Warning] Could not parse SMILES; atom map skipped: {save_path}")
        return None
    num_atoms = mol.GetNumAtoms()
    attn = np.asarray(atom_attn, dtype=float)
    attn = attn[:num_atoms] if len(attn) >= num_atoms else np.pad(attn, (0, num_atoms - len(attn)))

    svg_size = 500 if num_atoms <= 40 else (700 if num_atoms <= 80 else 900)
    norm = mcolors.Normalize(vmin=attn.min(), vmax=attn.max())
    cmap = matplotlib.colormaps[cmap_name]
    atom_colors = {i: tuple(cmap(norm(w))[:3]) for i, w in enumerate(attn)}
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", f"{atom.GetSymbol()}{atom.GetIdx()}")

    d = rdMolDraw2D.MolDraw2DSVG(svg_size, svg_size)
    opts = d.drawOptions()
    opts.clearBackground = True
    opts.backgroundColor = (1, 1, 1, 1)
    opts.addAtomIndices = False
    opts.minFontSize = 20
    if num_atoms > 40:
        opts.bondLineWidth = 1.5
        opts.multipleBondOffset = 0.15
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(atom_colors.keys()),
                                       highlightAtomColors=atom_colors)
    d.FinishDrawing()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(d.GetDrawingText())
    return save_path


def draw_heatmap(matrix, row_labels, atom_labels, title, save_path, cmap="Reds"):
    n_atoms = len(atom_labels)
    if n_atoms <= 30:
        fig_w, step = max(10, n_atoms * 0.45), 1
    elif n_atoms <= 80:
        fig_w, step = n_atoms * 0.35, 1
    elif n_atoms <= 150:
        fig_w, step = n_atoms * 0.28, 2
    else:
        fig_w, step = n_atoms * 0.22, 3
    xt = [l if i % step == 0 else "" for i, l in enumerate(atom_labels)]
    fig, ax = plt.subplots(figsize=(fig_w, max(3, len(row_labels) * 0.7)))
    sns.heatmap(matrix, cmap=cmap, cbar=True, xticklabels=xt, yticklabels=row_labels,
                linewidths=0.5, linecolor="gray", ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Atom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Protein / EC class", fontsize=12, fontweight="bold")
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_ec_attention_analysis(drug_name, drug_smiles, bacteria_name,
                              protein_ids, cross1_list, cross2_list,
                              kofam_tsv, fasta_path, save_dir,
                              user_proteins=None, top_fraction=TOP_FRACTION,
                              ko_ec_path=KO_EC_PATH, verbose=True):
    """
    Run protein ranking, EC sub-subclass enrichment, EC-averaged atom maps and
    user-defined protein maps for one drug-bacteria pair.

    protein_ids must be in the same order as the protein tokens given to the model.
    """
    os.makedirs(save_dir, exist_ok=True)
    user_proteins = [p for p in (user_proteins or [])]
    ko_to_ec = load_ko_to_ec(ko_ec_path)
    first_ko = load_first_ko(kofam_tsv)
    descriptions = load_fasta_descriptions(fasta_path)

    fold_attn, fold_rank, ens_c1 = aggregate_fold_attention(cross1_list, cross2_list)
    prot_df, top_n = build_protein_table(protein_ids, fold_attn, fold_rank, first_ko,
                                         ko_to_ec, descriptions, user_proteins, top_fraction)
    pct = int(round(top_fraction * 100))
    enr = ec_level3_enrichment(prot_df, top_n)

    mol, atom_labels = _atom_labels(drug_smiles, ens_c1.shape[1])
    n_heavy = mol.GetNumAtoms() if mol is not None else ens_c1.shape[1]

    prot_df.to_csv(os.path.join(save_dir, "protein_attention_ranking.csv"), index=False)
    enr.to_csv(os.path.join(save_dir, f"EC_enrichment_top{pct}pct.csv"), index=False)

    atom_value_cols = {"atom_index": np.arange(ens_c1.shape[1]), "atom_label": atom_labels}
    heat_rows, heat_labels = [], []

    # --- Representative EC classes (FE > 1 & P <= 0.05) ---
    # For each class: atom map averaged over its top-ranked member proteins,
    # plus the atom map of each member protein, so that the user can inspect
    # and select the EC class(es) of interest.
    sig = enr[enr[SIG_COL]] if not enr.empty else enr
    ec_root = os.path.join(save_dir, "EC_atom_maps")
    summary_rows, protein_rows = [], []
    for _, r in sig.iterrows():
        ec = r["EC_level3"]
        term = ec.replace("EC ", "")
        ec_dir = os.path.join(ec_root, _safe(ec))
        os.makedirs(ec_dir, exist_ok=True)
        members = prot_df[(prot_df["order"] <= top_n) &
                          prot_df["ec_level3"].apply(lambda s: term in s.split(";"))]
        idx = members["protein_index"].tolist()
        ec_attn = ens_c1[idx, :].mean(axis=0)
        mean_svg = draw_atom_attention(drug_smiles, ec_attn,
                                       os.path.join(ec_dir, f"atom_attention_{_safe(ec)}_mean.svg"))
        mean_rel = os.path.relpath(mean_svg, save_dir) if mean_svg else ""
        atom_value_cols[f"{ec} (mean of {len(idx)} top-ranked proteins)"] = ec_attn
        heat_rows.append(ec_attn[:n_heavy])
        heat_labels.append(f"{ec} (mean, n={len(idx)})")

        for _, mrow in members.iterrows():
            own = ens_c1[int(mrow["protein_index"]), :]
            p_svg = draw_atom_attention(drug_smiles, own,
                                        os.path.join(ec_dir, f"atom_attention_{_safe(mrow['protein_id'])}.svg"))
            atom_value_cols[f"{ec} | {mrow['protein_id']}"] = own
            protein_rows.append({
                "EC_level3": ec,
                "Fold_enrichment": r["Fold_enrichment"],
                "P_value(Fisher_one_sided)": r["P_value(Fisher_one_sided)"],
                "protein_id": mrow["protein_id"],
                "description": mrow["description"],
                "kegg_ko(first)": mrow["kegg_ko(first)"],
                "ec_numbers": mrow["ec_numbers"],
                "order": int(mrow["order"]),
                "mean_rank": mrow["mean_rank"],
                "user_defined": bool(mrow["user_defined"]),
                "Protein_atom_map": os.path.relpath(p_svg, save_dir) if p_svg else "",
                "EC_mean_atom_map": mean_rel,
            })

        summary_rows.append({
            "EC_level3": ec,
            "Fold_enrichment": r["Fold_enrichment"],
            "P_value(Fisher_one_sided)": r["P_value(Fisher_one_sided)"],
            "FDR_q_value(BH)": r["FDR_q_value(BH)"],
            "N_class_in_top": r["N_class_in_top"],
            "N_top": r["N_top"],
            "N_class_in_background": r["N_class_in_background"],
            "N_background": r["N_background"],
            "Protein_ids": "; ".join(members["protein_id"]),
            "Protein_descriptions": "; ".join(members["description"]),
            "Protein_full_EC": "; ".join(members["ec_numbers"]),
            "Protein_order": "; ".join(str(o) for o in members["order"]),
            "EC_mean_atom_map": mean_rel,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(save_dir, f"EC_candidates_top{pct}pct.csv"), index=False)
    pd.DataFrame(protein_rows).to_csv(
        os.path.join(save_dir, f"EC_candidate_proteins_top{pct}pct.csv"), index=False)

    # --- User-defined proteins: own EC + own atom map ---
    representative_terms = set(sig["EC_level3"]) if not sig.empty else set()
    user_rows = []
    for pid in user_proteins:
        hit = prot_df[prot_df["protein_id"] == pid]
        if hit.empty:
            print(f"[Warning] User-defined protein '{pid}' is not in the model input of "
                  f"{bacteria_name}; it is skipped for attention analysis.")
            continue
        h = hit.iloc[0]
        own_attn = ens_c1[int(h["protein_index"]), :]
        svg = draw_atom_attention(drug_smiles, own_attn,
                                  os.path.join(save_dir, f"atom_attention_user_{_safe(pid)}.svg"))
        atom_value_cols[f"{pid} (user-defined)"] = own_attn
        heat_rows.append(own_attn[:n_heavy])
        heat_labels.append(f"{pid} (user, #{int(h['order'])})")
        l3 = [] if h["ec_level3"] == "-" else [f"EC {t}" for t in h["ec_level3"].split(";")]
        user_rows.append({
            "protein_id": pid,
            "description": h["description"],
            "kegg_ko(first)": h["kegg_ko(first)"],
            "ec_numbers": h["ec_numbers"],
            "ec_level3": h["ec_level3"],
            "mean_rank": h["mean_rank"],
            "order": int(h["order"]),
            "n_input_proteins": len(prot_df),
            f"in_top{pct}pct": bool(h[f"in_top{pct}pct"]),
            "ec_level3_representative": "; ".join(t for t in l3 if t in representative_terms) or "-",
            "Atom_map": os.path.basename(svg) if svg else "",
        })
    user_df = pd.DataFrame(user_rows)
    if user_proteins:
        user_df.to_csv(os.path.join(save_dir, "user_defined_proteins.csv"), index=False)

    pd.DataFrame(atom_value_cols).to_csv(os.path.join(save_dir, "atom_attention_values.csv"), index=False)

    if heat_rows:
        draw_heatmap(np.vstack(heat_rows), heat_labels, atom_labels[:n_heavy],
                     f"{drug_name} - {bacteria_name}: atom attention",
                     os.path.join(save_dir, "atom_attention_heatmap.png"))

    if verbose:
        _print_summary(drug_name, bacteria_name, len(prot_df), top_n, pct, summary_df, user_df)

    return {"protein_table": prot_df, "enrichment": enr,
            "enriched_summary": summary_df, "user_proteins": user_df,
            "ensemble_cross1": ens_c1}


def _print_summary(drug_name, bacteria_name, n_prot, top_n, pct, summary_df, user_df):
    print(f"\n{'=' * 70}\n[Attention analysis] {drug_name} - {bacteria_name}\n{'=' * 70}")
    print(f"  Input proteins: {n_prot} | Top {pct}% (by mean fold rank): {top_n}")
    if summary_df.empty:
        print("  No EC sub-subclass met FE > 1 and P <= 0.05 in the top-ranked proteins.")
    else:
        print("  Representative EC sub-subclasses (one-sided Fisher's exact test, FE > 1 and P <= 0.05):")
        for _, r in summary_df.iterrows():
            print(f"   - {r['EC_level3']}: FE={r['Fold_enrichment']:.2f}, "
                  f"P={r['P_value(Fisher_one_sided)']:.4g}, "
                  f"{r['N_class_in_top']}/{r['N_class_in_background']} in top")
            ids = r["Protein_ids"].split("; ")
            descs = r["Protein_descriptions"].split("; ")
            for pid, d in zip(ids, descs):
                print(f"       {pid}  {d}")
    if user_df is not None and not user_df.empty:
        print("  User-defined proteins:")
        for _, r in user_df.iterrows():
            print(f"   - {r['protein_id']} ({r['description']}): KO={r['kegg_ko(first)']}, "
                  f"EC={r['ec_numbers']}, rank #{r['order']}/{r['n_input_proteins']}")


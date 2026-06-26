import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
save_dir = os.path.join(parent_dir, 'attention analysis')


def analyze_attention(drug_name, drug_smiles, target_protein, prot_dict,
                      cr1_att, cr2_att,
                      save_dir, target_prot_idx):

    print(f"\n{'='*60}\nAnalyzing: {drug_name}\n{'='*60}")

    cr1_avg = cr1_att.mean(dim=0).detach().cpu().numpy()
    cr2_avg = cr2_att.mean(dim=0).detach().cpu().numpy()

    idx_to_use = target_prot_idx
    if idx_to_use >= cr2_avg.shape[1]:
        print(f"[Error] target_prot_idx={idx_to_use} out of range "
              f"(max={cr2_avg.shape[1]-1})")
        return
    
    prot_keys_local = (list(prot_dict.keys())
                       if prot_dict and isinstance(prot_dict, dict) else [])
    def get_prot_name_local(idx):
        if idx < len(prot_keys_local):
            k = prot_keys_local[idx]
            return k.split('|')[-1] if '|' in k else k
        return f"Prot_{idx}"

    # Protein ranking
    prot_importance = cr2_avg.mean(axis=0)
    rank = (len(prot_importance)
            - np.argsort(np.argsort(prot_importance))[idx_to_use])
    
    print(f"\n[Protein Ranking]")
    print(f"  Target : {target_protein} (idx={idx_to_use})")
    print(f"  Score  : {prot_importance[idx_to_use]:.4f}")
    print(f"  Rank   : #{rank} / {len(prot_importance)}")

    print(f"  Total Avg Score : {prot_importance.mean():.4f}")

    sorted_idx = np.argsort(prot_importance)[::-1]
    table_data = []
    for r_i, si in enumerate(sorted_idx, start=1):
        name = get_prot_name_local(si)
        is_target = (si == idx_to_use)
        table_data.append({
            'Rank': r_i,
            'Protein': f"{name} (target)" if is_target else name,
            'Index': si,
            'Attention Score': f"{prot_importance[si]:.6f}",
            'Target': is_target
        })

    df_table = pd.DataFrame(table_data)
    df_table.to_csv(
        os.path.join(save_dir,
                     f"protein_importance_for_{drug_name}.csv"),
        index=False)
    
    # Attention Visualization
    atom_attn = cr1_avg[idx_to_use, :]
    mol = (Chem.MolFromSmiles(drug_smiles) if drug_smiles else None)
    n_atoms_mol = mol.GetNumAtoms() if mol else len(atom_attn)

    if len(atom_attn) >= n_atoms_mol:
        attn_for_atoms = atom_attn[:n_atoms_mol]
    else:
        attn_for_atoms = np.pad(atom_attn,
                                (0, n_atoms_mol - len(atom_attn)))
        
    atom_labels = ([f"{a.GetSymbol()}{a.GetIdx()}"
                    for a in mol.GetAtoms()]
                   if mol
                   else [f"A{i}" for i in range(len(attn_for_atoms))])
    n_atoms = len(atom_labels)

    cmap = "YlOrRd"

    if mol is not None:
        mol_noH = Chem.RemoveHs(mol)
        n_heavy = mol_noH.GetNumAtoms()

        heavy_attn = attn_for_atoms[:n_heavy]

        svg_size = 500 if n_heavy <= 40 else (700 if n_heavy <= 80 else 900)

        norm = colors.Normalize(vmin=heavy_attn.min(), vmax=heavy_attn.max())
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {i: tuple(mapper.to_rgba(w)[:3]) for i, w in enumerate(heavy_attn)}

        for atom in mol_noH.GetAtoms():
            atom.SetProp("atomNote", f"{atom.GetSymbol()}{atom.GetIdx()}")

        d = rdMolDraw2D.MolDraw2DSVG(svg_size, svg_size)
        opts = d.drawOptions()
        opts.clearBackground = True
        opts.backgroundColor = (1, 1, 1, 1)
        opts.addAtomIndices = False

        if n_heavy > 40:
            opts.bondLineWidth = 1.5
            opts.minFontSize = 10
            opts.multipleBondOffset = 0.15

        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol_noH,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors
        )
        d.FinishDrawing()
        svg_text = d.GetDrawingText()

        svg_path = os.path.join(save_dir, f'Attention Analysis for {drug_name}.svg')
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_text)

        # Top 5 + Target 
        top5_idx = list(np.argsort(prot_importance)[::-1][:5])
        heatmap_idx = list(top5_idx)
        if idx_to_use not in heatmap_idx:
            heatmap_idx.append(idx_to_use)

        sub_matrix = cr1_avg[heatmap_idx, :n_atoms_mol]
        y_labels = []
        for pi in heatmap_idx:
            name = get_prot_name_local(pi)
            r = (len(prot_importance)
                - np.argsort(np.argsort(prot_importance))[pi])
            if pi == idx_to_use:
                y_labels.append(f"★ {name} (#{r}, target)")
            else:
                y_labels.append(f"{name} (#{r})")

        # Dynamic sizing
        if n_atoms <= 30:
            fig_w3, tick_fs3, step3 = max(10, n_atoms * 0.45), 7, 1
        elif n_atoms <= 80:
            fig_w3, tick_fs3, step3 = n_atoms * 0.35, 6, 1
        elif n_atoms <= 150:
            fig_w3, tick_fs3, step3 = n_atoms * 0.28, 5, 2
        else:
            fig_w3, tick_fs3, step3 = n_atoms * 0.22, 4, 3

        display_labels3 = [l if i % step3 == 0 else ''
                        for i, l in enumerate(atom_labels)]
        fig_h3 = max(3, len(heatmap_idx) * 0.7)

        fig, ax = plt.subplots(figsize=(fig_w3, fig_h3))
        sns.heatmap(sub_matrix, cmap=cmap, cbar=True, annot=False,
                    xticklabels=display_labels3,
                    yticklabels=y_labels,
                    linewidths=0.5, linecolor='gray')
        ax.set_title(f'{drug_name} — Atom Attention(Top 5 Proteins + Target Protein)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Atom', fontsize=14, fontweight='bold')
        ax.set_ylabel('Protein', fontsize=14, fontweight='bold')
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                f'heatmap for {drug_name} with top 5 protein and target protein.png'),
                    dpi=300, bbox_inches='tight')
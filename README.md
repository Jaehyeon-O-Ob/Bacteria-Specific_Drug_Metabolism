# Bacteria-Specific_Drug_Metabolism
We developed the Multi Modal Bi-MHA model to predict bacteria-specific drug metabolism with attention-based interpretability map for insight mechanism between drug and bacteria

## 🛠 Environment Setup
This project uses a conda environment provided via `environment.yml`. You can create and activate the virtual environment with the required packages using the following commands:

```bash
conda env create -f environment.yml
conda activate emb_env
```
The only prerequisite is a working `conda` (Miniconda/Anaconda) installation. Every external tool the pipeline needs — including `hmmer`, and `ruby`/`parallel`/`git` (required by KofamScan and to download the models) — is installed by `environment.yml`, so no separate system setup is required.

## 🚀 How to Run Prediction
You can use the `prediction.py` script to predict the potential of drug metabolism between specific drugs and bacterial strains, and perform attention analysis.

### 1. Input Data Preparation
To run the prediction, you need a text file formatted like `sample1.txt`:
```text
drug: [Drug Name] (e.g., Diltiazem)
drug smiles: [Drug SMILES Code] (e.g., COc1ccc(C2Sc3ccccc3N(CC[NH+](C)C)C(=O)C2OC(C)=O)cc1)
bacterial strain name: [Bacterial Strain Name] (e.g., Bacteroides thetaiotaomicron VPI-5482)
target protein: [Protein ID(s), optional] (e.g., WP_008760980.1)
```
* **Note**: The protein sequence file for the bacterial strain specified in the input file (e.g., `[Strain Name]_proteins.fasta`) must exist in the `input/` directory.
* **`target protein` (optional)**: User-defined protein(s) of the bacterial strain of this record (comma-separated IDs are allowed, e.g. `target protein: WP_008760980.1, WP_011107642.1`). They are added to the model input, and their EC numbers and atom-attention maps are reported. Omit the line if you have no protein of interest. This is the same as the `--target_protein` / `--include_proteins` options, but applies only to the strain of this record.
* **Multiple pairs**: Write several records in one file, separated by a blank line. Each record can have its own `target protein` line.

### 2. Execution Command and Options
Run the script in your terminal as follows:
```bash
python prediction.py --input_txt [path_to_input_text_file] [additional_options]
```

**Arguments:**
- `--input_txt` (**Required**): Path to the input text file for prediction (e.g., `sample1.txt`).
- `--target_protein` (Optional): User-defined protein ID(s) (e.g., `WP_008760980.1`). They are automatically added to the prediction input, and their EC numbers and atom-attention maps are reported (see *Attention-based interpretation* below).
- `--include_proteins` (Optional): Comma-separated user-defined protein IDs to force-include in the prediction input (knowledge-based augmentation), e.g. `WP_011107642.1,WP_008760980.1`. Use this when you know a protein is metabolically relevant but it was dropped by the automatic KO filtering. Like `--target_protein`, each of these proteins is reported with its EC number and atom-attention map.

> **User-defined proteins (`target protein:` line, `--target_protein`, `--include_proteins`)** can be used alone or together. Command-line proteins are applied to every bacterial strain in the input file, whereas a `target protein:` line applies to the strain of its record; the proteins of a strain are the union of both (duplicate IDs are counted once). Because the protein input is built once per bacterial strain, records with the same strain share the same user-defined proteins. The IDs must exist in the strain's `*_proteins.fasta`; IDs that are not found are reported and ignored for that strain.
- `--esm_max_tokens` (Optional): Maximum tokens per batch for the ESM-C model embedding (Default: 2048).
- `--cpu_cores` (Optional): Number of parallel CPU cores to be used by KofamScan and XGBoost (Default: 8).

**Example:**
```bash
python prediction.py --input_txt sample1.txt
```
(`sample1.txt` already contains `target protein: WP_008760980.1`, so no protein option is needed.) Running this will execute KofamScan and the protein embedding process. The prediction results are saved as "prediction_results.csv", and the attention-based interpretation of each drug–bacteria pair is saved in `attention analysis/<drug>_<bacterial strain>/` and printed to the console.

> **Note:** The **first** run automatically downloads the required models and databases (ESM-C, MolE, and the KofamScan / KOfam database). This can take a while depending on your network; subsequent runs reuse the downloaded files and start immediately.

> **Note (KofamScan results are reused):** The KofamScan annotation of a bacterial strain is saved as `input/<Strain Name>_kegg_result.tsv` and is reused in all subsequent runs (`prediction.py` and `extract_representation.py`) for that strain, so KofamScan runs only once per strain. If you change the `*_proteins.fasta` of a strain, delete its `*_kegg_result.tsv` so that KofamScan is run again.

### 3. Prediction (5-fold ensemble)
The model consists of the five fold-specific multimodal Bi-MHA models and their paired XGBoost classifiers (`model/multi modal bi mha {1-5}.pth`, `model/xgb {1-5}.json`). For each drug–bacteria pair, every Bi-MHA model produces its own combined representation, which is classified by the XGBoost model of the same fold; the final probability is the mean of the five probabilities (Prediction = 1 if probability ≥ 0.5).

### 4. Attention-based interpretation
The interpretation follows the Case Study procedure of the paper [1]:
1. **Protein ranking** – For each fold-specific model, protein-level attention (drug atom tokens → bacterial protein tokens) is averaged over heads and atoms, and proteins are ranked within each fold. Proteins are ordered by their **mean rank across the five folds**.
2. **EC number enrichment** – Each protein is assigned the EC numbers of its first KofamScan KO hit (KEGG `link/enzyme/ko`, snapshot `utils/kegg_ko_to_ec (261105).tsv`). For the **top 10%** of the input proteins, every EC sub-subclass (e.g., EC 3.1.1) is tested with a one-sided Fisher's exact test against the complete input protein pool. **Every EC class with fold enrichment > 1 and P ≤ 0.05 is reported as a representative EC candidate**, together with its top-ranked proteins (protein ID and FASTA description), so that users can inspect the candidates and select the EC class(es) relevant to their drug.
3. **EC atom-attention maps** – Atom-level attention (bacterial protein tokens → drug atom tokens) is averaged over heads and over the five folds. For each representative EC class, (i) the attention rows of its top-ranked member proteins are averaged and drawn on the 2D structure, and (ii) the map of each member protein is drawn as well (darker red = higher relative attention).
4. **User-defined proteins** – For every protein given by a `target protein:` line or `--target_protein` / `--include_proteins`, its KO, EC number, mean rank and its own atom-attention map are reported.

> **Note:** P-values are unadjusted (as in the paper); Benjamini–Hochberg q-values are also given in `EC_enrichment_top10pct.csv` and `EC_candidates_top10pct.csv` for reference. The representative EC classes are hypotheses for experimental validation, not confirmed metabolic enzymes.

### 5. Result Example
`sample1.txt` is the diltiazem – *Bacteroides thetaiotaomicron* VPI-5482 pair, one of the three case studies of the paper [1] (diltiazem deacetylation; BT_4096 = `WP_008760980.1`). It is a high-reliability positive pair of the main dataset (Supplementary Table S1), and it was included in the training data of four of the five fold-specific models (it was held out only in fold 4). With the `target protein: WP_008760980.1` line (or `--target_protein WP_008760980.1`), the model input is the same as the input used in the paper (the same 672 proteins, ESM-C embeddings and KofamScan annotation; MolE atom embeddings and the KEGG pathway vector agree within floating-point precision, max. absolute difference < 1e-5), and the EC enrichment reproduces Table 3 of the paper (EC 3.1.1, fold enrichment 4.94, P = 0.005).

> **Note:** BT_4096 (`WP_008760980.1`) has no KO assigned by KofamScan and is therefore not retained by the automatic KO filtering. In the paper it was part of the model input as a knowledge-based protein; if the `target protein` line is removed (and no protein option is given), the input contains 671 proteins and the results differ slightly from the paper.

The example below was produced with `python prediction.py --input_txt sample1.txt`.

**Table of "prediction_results.csv"**

| Drug | Drug smiles code | Bacterial strain | Target protein | Prediction | Probability | Representative EC (top 10%, P<=0.05) |
| :--- | :--- | :--- | :--- | :--- | :-- | :-- |
| Diltiazem | COc1ccc(C2Sc3ccccc3N(CC[NH+](C)C)C(=O)C2OC(C)=O)cc1 | Bacteroides thetaiotaomicron VPI-5482 | WP_008760980.1 | 1 | 0.836 | EC 2.7.1; EC 3.1.1 |

- Prediction = 1 (True, The model predicts Drug is metabolized by Bacterial strain)
- Prediction = 0 (False, The model predicts Drug is not metabolized by Bacterial strain)

**Files in "attention analysis/Diltiazem_Bacteroides_thetaiotaomicron_VPI-5482/"**

| File | Content |
| :--- | :--- |
| `EC_candidates_top10pct.csv` | **Representative EC candidates** (FE > 1, P ≤ 0.05; one row per EC class) with counts, fold enrichment, P-value, top-ranked protein IDs / descriptions / full EC numbers and the path of the EC-averaged atom map |
| `EC_candidate_proteins_top10pct.csv` | One row per (EC class, protein) with description, KO, full EC number, rank and the paths of the protein-level and EC-averaged atom maps |
| `EC_atom_maps/EC_<EC>/atom_attention_EC_<EC>_mean.svg` | Atom-attention map averaged over the top-ranked proteins of the EC class |
| `EC_atom_maps/EC_<EC>/atom_attention_<protein>.svg` | Atom-attention map of each top-ranked protein of the EC class |
| `EC_enrichment_top10pct.csv` | Fisher's exact test result for every EC sub-subclass in the input protein pool |
| `protein_attention_ranking.csv` | All input proteins with description, first KO, EC numbers, fold-wise attention/rank, mean rank, top-10% flag, user-defined flag |
| `user_defined_proteins.csv` | KO, EC number, mean rank and representative-EC membership of each user-defined protein |
| `atom_attention_user_<protein>.svg` | Atom-attention map of each user-defined protein |
| `atom_attention_values.csv` | Atom-level attention values used for all maps |
| `atom_attention_heatmap.png` | Heatmap of the EC-averaged and user-defined protein atom attention |

**Representative EC candidates (top 10%, 68 of 672 input proteins)**

| EC class | Fold enrichment | P-value | Top-ranked proteins |
| :--- | :-- | :-- | :--- |
| EC 2.7.1 | 3.20 | 2.2e-4 | 11 kinases (see `EC_candidates_top10pct.csv`) |
| EC 3.1.1 | 4.94 | 0.0049 | WP_004295300.1 (lactonase family protein), WP_048696912.1 (6-phosphogluconolactonase), WP_011107784.1 (DUF452 family protein), WP_011109241.1 (lactonase family protein) |

**User-defined protein**: WP_008760980.1 (SGNH/GDSL hydrolase family protein) – no KO/EC assigned by KofamScan, mean-rank order #299 / 672.

<div align="center">
    <figure>
        <figcaption>
            <b>Atom-attention map averaged over the top-ranked EC 3.1.1 proteins (Diltiazem)</b><br>
            The darker the color, the higher the attention weight.<br>
        </figcaption>
        <img src="./attention analysis/Diltiazem_Bacteroides_thetaiotaomicron_VPI-5482/EC_atom_maps/EC_3.1.1/atom_attention_EC_3.1.1_mean.svg" alt="EC 3.1.1 atom attention for Diltiazem" width="500">
    </figure>
</div>

<div align="center">
    <figure>
        <figcaption>
            <b>Atom-attention map of the user-defined protein WP_008760980.1 (Diltiazem)</b><br>
        </figcaption>
        <img src="./attention analysis/Diltiazem_Bacteroides_thetaiotaomicron_VPI-5482/atom_attention_user_WP_008760980.1.svg" alt="WP_008760980.1 atom attention for Diltiazem" width="500">
    </figure>
</div>

<div align="center">
    <figure>
        <figcaption>
            <b>Heatmap of EC-averaged and user-defined protein atom attention (Diltiazem)</b><br>
        </figcaption>
        <img src="./attention analysis/Diltiazem_Bacteroides_thetaiotaomicron_VPI-5482/atom_attention_heatmap.png" alt="atom attention heatmap for Diltiazem" width="700">
    </figure>
</div>


## 🧬 Extracting Combined Representations
`extract_representation.py` extracts the combined representation (the input of the XGBoost classifier) for each drug–bacteria pair in the input text file:
```bash
python extract_representation.py --input_txt sample1.txt
```
- `--output_csv` (Optional): Base filename (Default: `combined_representations.csv`).
- `--target_protein`, `--include_proteins`, `--esm_max_tokens`, `--cpu_cores`: Same as `prediction.py`. User-defined proteins (including the `target protein:` line of the input file) are added in the same way as in `prediction.py`, so the extracted representations correspond to the same model input.

One file is saved per fold-specific model (`combined_representations_fold_1.csv` … `combined_representations_fold_5.csv`), each with the columns `drug`, `bacterial strain name`, `drug smiles code`, `dim_1` … `dim_512`. The representations are **not averaged across the five models**: the models were trained independently and their latent dimensions are not aligned (the same dimension index does not encode the same feature in different models), so each fold's representation should be used on its own (e.g., with the XGBoost model of the same fold).

## 📓 Example Notebook

The `Example_Prediction_and_Attention.ipynb` file included in the project is a Jupyter Notebook tutorial that allows you to visually experience the entire process of this prediction pipeline.

This tutorial notebook covers the following steps:
1. **Input Data Check**: Verifies the presence of the required `sample1.txt` file and the protein sequence files in the `input/` folder.
2. **Model Execution**: Guides you on how to run the pipeline within a notebook cell using the `!python prediction.py ...` command. 
    * *(Note: The first execution may take some time due to downloading the associated models and KofamScan database.)*
3. **Attention-Based Interpretation**: Shows how to load the representative EC candidates, the user-defined protein table, and the EC-level / protein-level atom-attention maps after the execution is complete.

---
### Dataset (Main / External Validation / Application)

For the main training, external validation, and application datasets, see the Supplementary section of the paper [1] (main dataset: Supplementary Table S1; application dataset: Supplementary Tables S7 and S8).

---
### Reference

[1] Park J, Kim SE, Yang H, Cho N-C, Kim H, Shin HK. Prediction of bacteria-specific drug metabolism using a cross-attention-based model. *Briefings in Bioinformatics* 2026;27(5):bbag578. https://doi.org/10.1093/bib/bbag578

---

#### 📥KOfam Database (KofamScan)

KofamScan annotates the bacterial proteins with KEGG Orthology (KO) numbers using two files: `ko_list` and the `profiles/` HMMs. On the first run these are downloaded automatically into `utils/kofam_db/` from a **fixed KOfam archive (`2026-01-01`)**, so that everyone annotates against the same database the model expects.

> **⚠️ Performance caveat:** We strongly recommend keeping the default `2026-01-01` archive. The model was trained and validated using KO annotations from this specific KOfam archive. Switching to a different release (`current` or another date) changes which KO numbers are assigned to the proteins, which in turn shifts the model's input features. As a result, predictions may differ from the reported results and **the model's performance is no longer guaranteed.** Use a different archive only if you understand and accept this trade-off.

If you would rather use the latest KOfam release, set the `KOFAM_ARCHIVE` environment variable before running the prediction:
```bash
# Use the newest ko_list / profiles
KOFAM_ARCHIVE=current python prediction.py --input_txt sample1.txt

# Or pin to a specific archive date listed at
# https://www.genome.jp/ftp/db/kofam/archives/
KOFAM_ARCHIVE=2024-01-01 python prediction.py --input_txt sample1.txt
```
To re-download a different version, delete the existing `utils/kofam_db/ko_list` and `utils/kofam_db/profiles/` first.
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
```
* **Note**: The protein sequence file for the bacterial strain specified in the input file (e.g., `[Strain Name]_proteins.fasta`) must exist in the `input/` directory.

### 2. Execution Command and Options
Run the script in your terminal as follows:
```bash
python prediction.py --input_txt [path_to_input_text_file] [additional_options]
```

**Arguments:**
- `--input_txt` (**Required**): Path to the input text file for prediction (e.g., `sample1.txt`).
- `--target_protein` (Optional): Specific target protein ID for attention analysis (e.g., `WP_008760980.1`). It is automatically added to the prediction input so its attention can be computed.
- `--include_proteins` (Optional): Comma-separated protein IDs to force-include in the prediction input (knowledge-based augmentation), e.g. `WP_011107642.1,WP_008760980.1`. Use this when you know a protein is metabolically relevant but it was dropped by the automatic KO filtering. These proteins are added to the model input but are **not** analyzed for attention (use `--target_protein` for that). The IDs must exist in the strain's `*_proteins.fasta`; unknown IDs are reported and ignored.
- `--esm_max_tokens` (Optional): Maximum tokens per batch for the ESM-C model embedding (Default: 2048).
- `--cpu_cores` (Optional): Number of parallel CPU cores to be used by KofamScan and XGBoost (Default: 8).

**Example:**
```bash
python prediction.py --input_txt sample1.txt --target_protein WP_008760980.1
```
Running this will execute KofamScan and the protein embedding process. The prediction results will be generated as "prediction_results.csv" and the attention map data will be saved in the `attention analysis/` folder and printed to the console.

> **Note:** The **first** run automatically downloads the required models and databases (ESM-C, MolE, and the KofamScan / KOfam database). This can take a while depending on your network; subsequent runs reuse the downloaded files and start immediately.

### 3. Result Example

**Table of "prediction_results.csv"**

| Drug | Drug smiles code | Bacterial strain | Prediction | Probability |
| :--- | :--- | :--- | :--- | :-- |
| Diltiazem | COc1ccc(C2Sc3ccccc3N(CC[NH+](C)C)C(=O)C2OC(C)=O)cc1 | Bacteroides thetaiotaomicron VPI-5482 | 1 | 0.617 |

- Prediction = 1 (True, The model predicts Drug is metabolized by Bacterial strain)
- Prediction = 0 (False, The model predicts Drug is not metabolized by Bacterial strain)

**Figures in "attention analysis"**

<div align="center">
    <figure>
        <figcaption>
            <b>Attention Analysis for Diltiazem</b><br>
            The darker the color, the higher the attention weight, indicating that the model predicts the region as a key metabolic site. <br>
        </figcaption>
        <img src="./attention analysis/Attention Analysis for Diltiazem.svg" alt="Attention Analysis for Diltiazem" width="500">
    </figure>
</div>

<div align="center">
    <figure>
        <figcaption>
            <b>Heatmap for Diltiazem with top 5 protein and target protein</b><br>
        <figcaption>
        <img src="./attention analysis/heatmap for Diltiazem with top 5 protein and target protein.png" alt="heatmap for Diltiazem with top 5 protein and target protein" width="600">
        </figure>
</div>


## 📓 Example Notebook

The `Example_Prediction_and_Attention.ipynb` file included in the project is a Jupyter Notebook tutorial that allows you to visually experience the entire process of this prediction pipeline.

This tutorial notebook covers the following steps:
1. **Input Data Check**: Verifies the presence of the required `sample1.txt` file and the protein sequence files in the `input/` folder.
2. **Model Execution**: Guides you on how to run the pipeline within a notebook cell using the `!python prediction.py ...` command. 
    * *(Note: The first execution may take some time due to downloading the associated models and KofamScan database.)*
3. **Attention Analysis Visualization**: Shows how to load the output attention results (e.g., Heatmap images) after the execution is complete, and guides you in visually interpreting the interaction mechanism between the drug and the specific target bacterial protein.

---
### Dataset (Main / External Validation / Application)

For the main training, external validation, and application datasets, see the Supplementary section of the paper [1].

---
### Reference

[1] it will be updated.

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
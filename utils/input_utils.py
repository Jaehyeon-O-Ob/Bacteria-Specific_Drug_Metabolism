"""
input_utils.py
==============

Shared input handling for ``prediction.py`` and ``extract_representation.py``.

Input text format (records separated by a blank line)::

    drug: Diltiazem
    drug smiles: COc1ccc(C2Sc3ccccc3N(CC[NH+](C)C)C(=O)C2OC(C)=O)cc1
    bacterial strain name: Bacteroides thetaiotaomicron VPI-5482
    target protein: WP_008760980.1          # optional, comma-separated IDs allowed

User-defined proteins
---------------------
User-defined proteins can be given in two ways, and both can be combined:

* ``target protein:`` line in the input text file - applies to the bacterial
  strain of that record.
* ``--target_protein`` / ``--include_proteins`` command-line options - apply to
  every bacterial strain in the input file (IDs that are not present in a
  strain's FASTA are reported and ignored for that strain).

The protein set given to the model is built once per bacterial strain, so the
user-defined proteins of a strain are the union of the command-line proteins
and the ``target protein`` entries of all records with that strain. Duplicate
IDs are removed.
"""

import pandas as pd


def _split_ids(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return [p.strip() for p in str(text).split(',') if p.strip()]


def parse_input_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    current_record = {}

    for line in lines:
        line = line.strip()
        if not line:
            if current_record:
                records.append(current_record)
                current_record = {}
            continue

        lower_line = line.lower()
        value = line.split(':', 1)[1].strip() if ':' in line else ''
        if lower_line.startswith('drug smiles'):
            current_record['drug smiles code'] = value
        elif lower_line.startswith('drug:') or lower_line.startswith('drug name:'):
            current_record['drug'] = value
        elif lower_line.startswith('bacterial strain'):
            current_record['bacterial strain name'] = value
        elif lower_line.startswith('target protein'):
            # 'target protein:' or 'target proteins:'; comma-separated IDs allowed
            current_record['target protein'] = ', '.join(_split_ids(value))

    if current_record:
        records.append(current_record)

    # A block that only contains a 'target protein:' line (e.g. written after a
    # blank line) is attached to the preceding record.
    merged = []
    for rec in records:
        if set(rec.keys()) == {'target protein'} and merged:
            prev = merged[-1]
            ids = _split_ids(prev.get('target protein', '')) + _split_ids(rec['target protein'])
            prev['target protein'] = ', '.join(dict.fromkeys(ids))
            print(f"[Info] 'target protein: {rec['target protein']}' was attached to the preceding "
                  f"record ({prev.get('drug', '?')} - {prev.get('bacterial strain name', '?')}).")
        else:
            merged.append(rec)
    records = merged

    required = ['drug', 'drug smiles code', 'bacterial strain name']
    for i, rec in enumerate(records, start=1):
        missing = [k for k in required if not rec.get(k)]
        if missing:
            raise ValueError(f"Record {i} in '{file_path}' is missing required key(s): "
                             f"{', '.join(missing)} (record: {rec})")

    df = pd.DataFrame(records)
    if 'target protein' not in df.columns:
        df['target protein'] = ''
    df['target protein'] = df['target protein'].fillna('')
    return df


def collect_user_proteins(pred_df, bacteria_col, target_protein=None, include_proteins=None):
    """
    Returns {bacterial strain: [protein id, ...]} combining the command-line
    proteins (all strains) and the per-record 'target protein' entries.
    """
    cli = []
    for pid in _split_ids(include_proteins) + _split_ids(target_protein):
        if pid not in cli:
            cli.append(pid)

    user = {}
    for bac in pred_df[bacteria_col].unique().tolist():
        prots = list(cli)
        for txt in pred_df.loc[pred_df[bacteria_col] == bac, 'target protein']:
            for pid in _split_ids(txt):
                if pid not in prots:
                    prots.append(pid)
        user[bac] = prots
    return user

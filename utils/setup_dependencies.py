"""
setup_dependencies.py
=====================

Automatically downloads every external dependency that ``prediction.py`` needs,
so a user only has to run ``prediction.py`` and everything else is fetched on
the first run.

What it sets up
---------------
1. **ESM-C source code** (``utils/esm_official``) - cloned from GitHub. The model
   weights themselves are downloaded automatically from Hugging Face at runtime
   by ``ESMC.from_pretrained(...)`` (see ``bacteria_rep_generate.py``).
2. **MolE source code** (``utils/mole_antimicrobial_potential``) - cloned from
   GitHub, plus the pre-trained model binary ``model.pth`` downloaded from
   Zenodo (record 10803099).
3. **KofamScan + KOfam database** (``utils/kofam_db``) - the ``kofam_scan-1.3.0``
   executable, and the ``ko_list`` / ``profiles`` HMMs downloaded from the
   GenomeNet FTP archive.

Everything is idempotent: anything already present on disk is left untouched.
The first run can therefore take a long time (large downloads), but subsequent
runs return immediately.
"""

import os
import ssl
import gzip
import shutil
import tarfile
import subprocess
import urllib.request

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- External sources -------------------------------------------------------
ESM_REPO_URL = "https://github.com/Biohub/esm"
# Pin ESM-C to v3.2.3. Later versions (the "oss sync" release) switched ESMC
# weight loading to huggingface_hub.load_torch_model(), which is incompatible
# with the data/weights/*.pth layout of the published esmc-600m checkpoint and
# fails with "does not contain a valid checkpoint". v3.2.3 loads the weights
# directly (torch.load) and matches the version this project was validated with.
ESM_REPO_REF = "v3.2.3"
MOLE_REPO_URL = "https://github.com/rolayoalarcon/mole_antimicrobial_potential"

# MolE pre-trained model binary (~800 MB, Zenodo record 10803099)
MOLE_MODEL_URL = "https://zenodo.org/records/10803099/files/model.pth?download=1"

# KofamScan executable (the annotation tool itself)
KOFAM_SCAN_URL = "https://www.genome.jp/ftp/tools/kofam_scan/kofam_scan-1.3.0.tar.gz"

# KOfam database snapshot.
#
# Pinned to a fixed archive date so that every user annotates against the exact
# same database the model expects. To use a newer release instead, set the
# KOFAM_ARCHIVE environment variable before running prediction.py to either:
#   * another date listed at https://www.genome.jp/ftp/db/kofam/archives/
#     (e.g. KOFAM_ARCHIVE=2024-01-01), or
#   * "current" to use the latest ko_list/profiles at
#     https://www.genome.jp/ftp/db/kofam/
KOFAM_ARCHIVE = os.environ.get("KOFAM_ARCHIVE", "2026-01-01")


def _kofam_base_url():
    if KOFAM_ARCHIVE.lower() == "current":
        return "https://www.genome.jp/ftp/db/kofam"
    return f"https://www.genome.jp/ftp/db/kofam/archives/{KOFAM_ARCHIVE}"


# --- Helpers ----------------------------------------------------------------
def _run(cmd):
    print(f"[setup] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _download(url, dest):
    """Download ``url`` to ``dest`` using wget/curl if available, otherwise the
    Python standard library. SSL verification is relaxed because genome.jp and
    some Zenodo mirrors occasionally serve certificates that fail strict checks
    in fresh conda environments."""
    tmp = dest + ".part"
    if shutil.which("wget"):
        _run(["wget", "--no-check-certificate", "-O", tmp, url])
    elif shutil.which("curl"):
        _run(["curl", "-fkL", "-o", tmp, url])
    else:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        print(f"[setup] downloading {url}")
        with urllib.request.urlopen(url, context=ctx) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
    os.replace(tmp, dest)


def _fetch_repo(repo_url, dest, ref=None):
    """Make ``dest`` contain a fresh checkout of ``repo_url``.

    ``ref`` optionally pins a tag/branch. Uses ``git clone`` when git is
    available, otherwise falls back to downloading the source tarball from
    GitHub - so git is not a hard requirement for a new user."""
    if os.path.isdir(dest):
        shutil.rmtree(dest)

    if shutil.which("git"):
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [repo_url, dest]
        _run(cmd)
        return

    # Fallback: download the source tarball (no git needed).
    print("[setup] 'git' not found; downloading source tarball instead...")
    if ref:
        tar_url = repo_url.rstrip("/") + "/archive/refs/tags/" + ref + ".tar.gz"
    else:
        tar_url = repo_url.rstrip("/") + "/archive/HEAD.tar.gz"
    tmp_tgz = dest + ".src.tar.gz"
    extract_dir = dest + ".extract"
    _download(tar_url, tmp_tgz)
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)
    with tarfile.open(tmp_tgz) as tar:
        tar.extractall(extract_dir)
    # GitHub tarballs wrap everything in a single top-level directory.
    entries = [os.path.join(extract_dir, e) for e in os.listdir(extract_dir)]
    top = entries[0] if len(entries) == 1 and os.path.isdir(entries[0]) else extract_dir
    shutil.move(top, dest)
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.remove(tmp_tgz)


# --- Individual dependencies ------------------------------------------------
def setup_esm():
    """Download the ESM-C source code. Weights are fetched from Hugging Face on
    first model use, not here."""
    esm_dir = os.path.join(CURRENT_DIR, "esm_official")
    if os.path.exists(os.path.join(esm_dir, "esm", "models", "esmc.py")):
        return
    print("[setup] Downloading ESM-C source code (this can take a while)...")
    _fetch_repo(ESM_REPO_URL, esm_dir, ref=ESM_REPO_REF)


def setup_mole():
    """Download the MolE source code and the pre-trained model binary."""
    mole_dir = os.path.join(CURRENT_DIR, "mole_antimicrobial_potential")
    if not os.path.exists(
        os.path.join(mole_dir, "workflow", "models", "ginet_concat.py")
    ):
        print("[setup] Downloading MolE source code (this can take a while)...")
        _fetch_repo(MOLE_REPO_URL, mole_dir)

    model_path = os.path.join(
        mole_dir,
        "pretrained_model",
        "model_ginconcat_btwin_100k_d8000_l0.0001",
        "model.pth",
    )
    if not os.path.exists(model_path):
        print("[setup] Downloading MolE pre-trained model (~800 MB, this can take a while)...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        _download(MOLE_MODEL_URL, model_path)


def setup_kofam_db():
    """Download the KofamScan tool and the KOfam ko_list / profiles database."""
    kofam_dir = os.path.join(CURRENT_DIR, "kofam_db")
    os.makedirs(kofam_dir, exist_ok=True)

    # 1. KofamScan executable
    scan_dir = os.path.join(kofam_dir, "kofam_scan-1.3.0")
    if not os.path.exists(os.path.join(scan_dir, "exec_annotation")):
        print("[setup] Downloading KofamScan tool...")
        tgz = os.path.join(kofam_dir, "kofam_scan-1.3.0.tar.gz")
        _download(KOFAM_SCAN_URL, tgz)
        with tarfile.open(tgz) as tar:
            tar.extractall(kofam_dir)
        os.remove(tgz)
        # Provide a working config.yml (hmmsearch is resolved from $PATH).
        cfg = os.path.join(scan_dir, "config.yml")
        tmpl = os.path.join(scan_dir, "config-template.yml")
        if not os.path.exists(cfg) and os.path.exists(tmpl):
            shutil.copy(tmpl, cfg)

    base = _kofam_base_url()

    # 2. ko_list
    ko_list = os.path.join(kofam_dir, "ko_list")
    if not os.path.exists(ko_list):
        print(f"[setup] Downloading KOfam ko_list ({KOFAM_ARCHIVE})...")
        gz = ko_list + ".gz"
        _download(f"{base}/ko_list.gz", gz)
        with gzip.open(gz, "rb") as fin, open(ko_list, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(gz)

    # 3. profiles
    profiles = os.path.join(kofam_dir, "profiles")
    if not os.path.isdir(profiles) or not os.listdir(profiles):
        print(f"[setup] Downloading KOfam profiles ({KOFAM_ARCHIVE}, large, this can take a while)...")
        tgz = os.path.join(kofam_dir, "profiles.tar.gz")
        _download(f"{base}/profiles.tar.gz", tgz)
        with tarfile.open(tgz) as tar:
            tar.extractall(kofam_dir)
        os.remove(tgz)


def ensure_all_dependencies():
    """Make sure every external dependency required by prediction.py is present.

    Safe to call on every run: already-installed dependencies are skipped."""
    setup_esm()
    setup_mole()
    setup_kofam_db()


if __name__ == "__main__":
    ensure_all_dependencies()
    print("[setup] All dependencies are ready.")

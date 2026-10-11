"""
==========================================================================
 Protein Degradation Prediction via Custom Edge-Conditioned WL Graph Kernel
==========================================================================

pip install numpy pandas scikit-learn biopython tqdm

Author  : Expert Computational Biologist / ML Engineer
Problem : Predict thermodynamic stability of therapeutic proteins from their
          3-D Residue Contact Networks (RCNs) using ΔΔG data from FireProtDB.
Kernel  : Custom Edge-Conditioned Weisfeiler-Lehman (WL) Graph Kernel —
          implemented entirely from scratch with pure Python + NumPy.
          No grakel or any other graph-kernel library is used.

Pipeline overview
-----------------
1. Mock FireProtDB DataFrame  →  binary stability labels
2. Biopython PDB fetch        →  Cα coordinates + residue names
3. RCN construction           →  node physicochemical labels + distance-bin edges
4. Custom WL kernel (scratch) →  N×N precomputed similarity matrix
5. SVC (precomputed kernel)   →  classification report
==========================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io
import re
import sys
import time
import warnings
import hashlib
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from tqdm import tqdm

# Biopython imports — for PDB fetching & parsing
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa

# Scikit-learn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Contact threshold: any Cα–Cα pair within 8.0 Å is an edge in the RCN.
CONTACT_THRESHOLD_A = 8.0  # ångströms

# Distance bins (ångströms) — encode bond/contact intimacy as an edge label.
#   close   < 4 Å  → strong van-der-Waals / hydrogen-bond range
#   medium  4–6 Å  → typical side-chain contact
#   far     6–8 Å  → long-range electrostatic / hydrophobic contact
DIST_BINS = [
    (0.0, 4.0, "close"),
    (4.0, 6.0, "medium"),
    (6.0, 8.0, "far"),
]

# ── Physicochemical property labels (h=0 node features) ──────────────────────
# Each standard amino acid is mapped to a two-part string:
#   "<hydrophobicity>_<charge>"
# This encodes the most structurally/energetically relevant residue properties.
AA_PHYSCHEM: Dict[str, str] = {
    # Hydrophobic, neutral
    "ALA": "Hydrophobic_Neutral",
    "VAL": "Hydrophobic_Neutral",
    "ILE": "Hydrophobic_Neutral",
    "LEU": "Hydrophobic_Neutral",
    "MET": "Hydrophobic_Neutral",
    "PHE": "Hydrophobic_Neutral",
    "TRP": "Hydrophobic_Neutral",
    "PRO": "Hydrophobic_Neutral",
    # Hydrophilic, neutral (polar uncharged)
    "SER": "Hydrophilic_Neutral",
    "THR": "Hydrophilic_Neutral",
    "CYS": "Hydrophilic_Neutral",
    "TYR": "Hydrophilic_Neutral",
    "ASN": "Hydrophilic_Neutral",
    "GLN": "Hydrophilic_Neutral",
    "GLY": "Hydrophilic_Neutral",
    # Hydrophilic, positive (basic)
    "LYS": "Hydrophilic_Positive",
    "ARG": "Hydrophilic_Positive",
    "HIS": "Hydrophilic_Positive",
    # Hydrophilic, negative (acidic)
    "ASP": "Hydrophilic_Negative",
    "GLU": "Hydrophilic_Negative",
    # Default for non-standard residues
    "UNK": "Unknown_Neutral",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  REAL FIREPROTDB LOADER  +  BINARY LABELS
# ══════════════════════════════════════════════════════════════════════════════

def load_fireprotdb(csv_path: str,
                    max_proteins: int = 50,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Load the real FireProtDB CSV export and return a clean DataFrame.

    FireProtDB schema notes (from the actual export)
    -------------------------------------------------
    - Column names are UPPERCASE (DDG, WWPDB, …).
    - DDG   : ΔΔG in kcal/mol — our target.  Many rows are NaN.
    - WWPDB : PDB accession(s) for the wild-type structure.
              Can be comma-separated (e.g. "1YGW, 1RN1") when multiple
              crystal forms exist; we take the first clean 4-char code.
    - ~5.4M total rows, but only ~7,977 have both DDG and WWPDB.

    Strategy
    --------
    1. Load only the two needed columns to avoid reading 1.7 GB into RAM.
    2. Drop rows missing DDG or WWPDB.
    3. Parse the first valid 4-character PDB code from WWPDB.
    4. Deduplicate: one representative row per unique PDB ID
       (median DDG of all mutations on that protein).
       This is intentional — the WL kernel operates on the STRUCTURE,
       so all mutations on the same PDB chain produce the same graph.
       We keep one label per protein so the kernel matrix is meaningful.
    5. Cap at max_proteins (default 50) for tractable kernel computation.
       The full N×N matrix for 175 proteins × 20 WL iterations is fast,
       but 175² = 30,625 entries → still very manageable if desired.

    Parameters
    ----------
    csv_path     : str  — path to the FireProtDB CSV file.
    max_proteins : int  — maximum number of unique proteins to use.
                          Set to None to use all (~175 unique proteins).
    random_state : int  — seed for reproducible sub-sampling.

    Returns
    -------
    pd.DataFrame with columns: pdb_id, ddG, label
    """
    print(f"[FireProtDB] Loading CSV: {csv_path}")
    print(f"[FireProtDB] Reading columns DDG + WWPDB …")

    # Read only the two relevant columns — avoids loading the full 1.7 GB
    df_raw = pd.read_csv(
        csv_path,
        usecols=["DDG", "WWPDB"],
        low_memory=False,
    )
    print(f"[FireProtDB] Raw rows: {len(df_raw):,}")

    # ── Keep only rows with both DDG and WWPDB ────────────────────────────
    df_raw = df_raw.dropna(subset=["DDG", "WWPDB"])
    print(f"[FireProtDB] Rows with DDG + WWPDB: {len(df_raw):,}")

    # ── Parse first valid 4-character PDB code from WWPDB ─────────────────
    # WWPDB can be "1WQ5" or "1YGW, 1RN1, 9RNT" — we take the first token
    # that matches the 4-alphanumeric PDB format.
    def extract_pdb_id(wwpdb_str: str) -> Optional[str]:
        # Split on comma/space, strip whitespace, keep first 4-char match
        for token in re.split(r"[,\s]+", str(wwpdb_str)):
            token = token.strip().upper()
            if re.fullmatch(r"[A-Z0-9]{4}", token):
                return token
        return None

    df_raw["pdb_id"] = df_raw["WWPDB"].apply(extract_pdb_id)
    df_raw = df_raw.dropna(subset=["pdb_id"])

    # ── Rename and cast DDG ───────────────────────────────────────────────
    df_raw = df_raw.rename(columns={"DDG": "ddG"})
    df_raw["ddG"] = pd.to_numeric(df_raw["ddG"], errors="coerce")
    df_raw = df_raw.dropna(subset=["ddG"])

    # ── Deduplicate: one row per unique PDB ID ────────────────────────────
    # Aggregate all mutations on the same protein → median ΔΔG.
    # This gives each protein a single representative stability label
    # that reflects its overall mutational landscape.
    df_protein = (
        df_raw.groupby("pdb_id", as_index=False)["ddG"]
        .median()
        .rename(columns={"ddG": "ddG"})
    )
    print(f"[FireProtDB] Unique proteins after dedup: {len(df_protein)}")

    # ── Binary label ──────────────────────────────────────────────────────
    df_protein["label"] = (df_protein["ddG"] >= 0).astype(int)

    # ── Sub-sample if needed ──────────────────────────────────────────────
    if max_proteins is not None and len(df_protein) > max_proteins:
        # Stratified sample to preserve class balance
        stable   = df_protein[df_protein["label"] == 1]
        unstable = df_protein[df_protein["label"] == 0]
        n_stable   = min(len(stable),   max_proteins // 2)
        n_unstable = min(len(unstable), max_proteins - n_stable)
        df_protein = pd.concat([
            stable.sample(n_stable,   random_state=random_state),
            unstable.sample(n_unstable, random_state=random_state),
        ]).reset_index(drop=True)
        print(f"[FireProtDB] Sampled to {len(df_protein)} proteins "
              f"(max_proteins={max_proteins}).")

    df_protein = df_protein[["pdb_id", "ddG", "label"]].reset_index(drop=True)

    print(f"[FireProtDB] Final: {len(df_protein)} records | "
          f"Stable(ΔΔG≥0)={df_protein['label'].sum()} | "
          f"Unstable(ΔΔG<0)={(df_protein['label']==0).sum()}")
    return df_protein


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BIOPYTHON: PDB FETCH  +  Cα EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def fetch_and_parse_pdb(pdb_id: str,
                        cache: Dict[str, Optional[dict]] = {},
                        pdb_dir: str = "./pdb_files") -> Optional[dict]:
    """
    Download a PDB structure and extract Cα coordinates + residue names.

    Strategy
    --------
    Primary  : Direct urllib download from RCSB (https://files.rcsb.org).
               This avoids any Biopython version dependency on PDBList's
               ever-changing kwarg names (file_type vs file_format, etc.).
    Fallback : Bio.PDB.PDBList with the correct kwarg for the installed
               Biopython version (file_format in ≥1.78, file_type in older).

    Parameters
    ----------
    pdb_id  : str   — 4-character PDB accession (case-insensitive).
    cache   : dict  — in-process memoisation; each structure fetched once.
    pdb_dir : str   — local directory to cache downloaded .pdb files.

    Returns
    -------
    dict with keys:
        'residues'    : list[str]   — three-letter residue names
        'ca_coords'   : np.ndarray  — shape (N, 3), ångströms
    or None on failure.
    """
    import os
    import urllib.request

    pdb_id = pdb_id.upper()
    if pdb_id in cache:
        return cache[pdb_id]

    os.makedirs(pdb_dir, exist_ok=True)
    local_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    parser     = PDBParser(QUIET=True)
    structure  = None

    # ── Primary: direct RCSB download via urllib ─────────────────────────
    # RCSB keeps canonical PDB-format files at this URL.
    # urllib is stdlib — no extra deps, no version drift.
    if not os.path.isfile(local_path):
        rcsb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            print(f"  [RCSB] Downloading {pdb_id} …", end=" ", flush=True)
            urllib.request.urlretrieve(rcsb_url, local_path)
            print("done.")
        except Exception as exc:
            print(f"\n  [WARN] urllib download failed for {pdb_id}: {exc}")
            # ── Fallback: Bio.PDB.PDBList ─────────────────────────────────
            # Biopython ≥1.78 renamed `file_type` → `file_format`.
            # We try both kwargs so the code works across versions.
            pdb_list = PDB.PDBList(verbose=False)
            try:
                local_path = pdb_list.retrieve_pdb_file(
                    pdb_id, file_format="pdb", pdir=pdb_dir, overwrite=False
                )
            except TypeError:
                try:
                    local_path = pdb_list.retrieve_pdb_file(   # older API
                        pdb_id, file_type="pdb", pdir=pdb_dir
                    )
                except Exception as exc2:
                    print(f"  [WARN] PDBList fallback also failed: {exc2}")
                    cache[pdb_id] = None
                    return None
            except Exception as exc2:
                print(f"  [WARN] PDBList fallback failed: {exc2}")
                cache[pdb_id] = None
                return None
    else:
        print(f"  [CACHE] Using cached {pdb_id}.pdb")

    # ── Parse the downloaded file ─────────────────────────────────────────
    try:
        structure = parser.get_structure(pdb_id, local_path)
    except Exception as exc:
        print(f"  [WARN] Could not parse {pdb_id}: {exc}")
        cache[pdb_id] = None
        return None

    residues_out: List[str]            = []
    coords_out:   List[np.ndarray]     = []

    # Iterate model → chain → residue; keep only standard AA with a Cα atom
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue          # skip HETATM, water, ligands
                try:
                    ca_atom = residue["CA"]
                except KeyError:
                    continue          # skip residues missing Cα (e.g. GLY in some files)

                resname = residue.get_resname().strip()
                residues_out.append(resname)
                coords_out.append(ca_atom.get_vector().get_array())

        break  # use only the first biological model

    if len(residues_out) < 2:
        print(f"  [WARN] {pdb_id}: fewer than 2 Cα atoms found — skipping.")
        cache[pdb_id] = None
        return None

    result = {
        "residues"  : residues_out,
        "ca_coords" : np.array(coords_out, dtype=np.float32),
    }
    cache[pdb_id] = result
    print(f"  [PDB] {pdb_id}: {len(residues_out)} residues | "
          f"{np.array(coords_out).shape[0]} Cα atoms extracted.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2b. SYNTHETIC PDB FALLBACK  (used when RCSB is unreachable)
# ══════════════════════════════════════════════════════════════════════════════

# Canonical sequences for the five demo proteins (UniProt/RCSB reference)
_KNOWN_SEQUENCES: Dict[str, str] = {
    "1UBQ": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "1L2Y": "NLYIQWLKDGGPSSGRPPPS",
    "2LZM": "NIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
    "1VII": "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    "1CRN": "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
}

# 1-letter → 3-letter amino acid code map
_ONE_TO_THREE: Dict[str, str] = {
    'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
    'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
    'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
    'W':'TRP','Y':'TYR',
}


def generate_synthetic_pdb_data(pdb_id: str, seed: int = 0) -> dict:
    """
    Generate synthetic but structurally plausible Cα coordinates for a protein.

    This fallback is used when RCSB is unreachable (e.g. offline/sandboxed
    environments).  It produces realistic mock data by:

    1. Using the known amino acid sequence for each demo PDB ID (if available),
       or generating a random sequence otherwise.
    2. Placing Cα atoms along an idealised α-helix backbone
       (rise = 1.5 Å/residue, radius = 2.3 Å, pitch = 100°/residue)
       — this is the dominant secondary structure in globular proteins.
    3. Adding Gaussian noise (σ = 0.8 Å) to break perfect symmetry and
       introduce realistic variation in edge distance bins.

    The resulting RCN is structurally distinct per protein because each
    protein has a different sequence (different h=0 node labels) and a
    different random seed (different edge-bin distribution).

    Parameters
    ----------
    pdb_id : str   — used to select a known sequence and set the RNG seed.
    seed   : int   — extra seed offset for reproducibility.

    Returns
    -------
    dict with keys 'residues' (list[str]) and 'ca_coords' (np.ndarray, N×3).
    """
    aa3 = [k for k in AA_PHYSCHEM.keys() if k != "UNK"]

    # Use known sequence if available, else sample randomly
    seq1 = _KNOWN_SEQUENCES.get(pdb_id.upper(), None)
    if seq1 is not None:
        residues = [_ONE_TO_THREE.get(c, 'ALA') for c in seq1 if c.isalpha()]
    else:
        rng_seq  = np.random.default_rng(seed + abs(hash(pdb_id)) % 10000)
        n_rand   = int(rng_seq.integers(30, 80))
        residues = [aa3[i] for i in rng_seq.integers(0, len(aa3), size=n_rand)]

    n   = len(residues)
    rng = np.random.default_rng(seed + abs(hash(pdb_id)) % 10000)

    # Idealised α-helix Cα trace
    # Parameters: rise=1.5 Å/residue, radius=2.3 Å, 100° rotation/residue
    t      = np.arange(n, dtype=float)
    angle  = np.deg2rad(100.0) * t
    x      = 2.3 * np.cos(angle)
    y      = 2.3 * np.sin(angle)
    z      = 1.5 * t
    coords = np.stack([x, y, z], axis=1)

    # Add Gaussian noise to get realistic distance-bin variation
    coords += rng.normal(scale=0.8, size=coords.shape)

    print(f"  [SYNTH] {pdb_id}: generated {n}-residue synthetic protein backbone.")
    return {"residues": residues, "ca_coords": coords.astype(np.float32)}


def get_pdb_data(pdb_id: str,
                 cache: Dict[str, Optional[dict]] = {},
                 pdb_dir: str = "./pdb_files",
                 allow_synthetic: bool = True) -> Optional[dict]:
    """
    Unified data-access router:
      1. Try to fetch the real PDB structure from RCSB via fetch_and_parse_pdb.
      2. If network is unavailable AND allow_synthetic=True, fall back to
         generate_synthetic_pdb_data (for demos / offline runs).
      3. If both fail, return None.

    Set allow_synthetic=False when running with a real FireProtDB CSV to ensure
    you never silently mix real and synthetic graphs.
    """
    key = pdb_id.upper()
    if key in cache and cache[key] is not None:
        return cache[key]

    result = fetch_and_parse_pdb(pdb_id, cache=cache, pdb_dir=pdb_dir)
    if result is not None:
        return result

    if allow_synthetic:
        print(f"  [FALLBACK] Using synthetic Cα data for {pdb_id} "
              f"(set allow_synthetic=False for production runs).")
        result = generate_synthetic_pdb_data(pdb_id)
        cache[key] = result
        return result

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RESIDUE CONTACT NETWORK (RCN) CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def get_distance_bin(dist: float) -> str:
    """
    Map a Cα–Cα Euclidean distance (ångströms) to a discrete bin label.

    Bins
    ----
    close  : dist < 4 Å   — strongest contacts, near-covalent geometry
    medium : 4 ≤ dist < 6 — typical residue–residue side-chain contacts
    far    : 6 ≤ dist < 8 — weak long-range contacts at the contact threshold

    These bins become edge features in the WL aggregation step, letting the
    kernel distinguish structurally distinct motifs that share the same
    topology but differ in contact intimacy.
    """
    for lo, hi, label in DIST_BINS:
        if lo <= dist < hi:
            return label
    return "far"   # fallback (distance == exactly 8.0 is excluded by threshold)


def build_rcn(pdb_data: dict) -> dict:
    """
    Build a Residue Contact Network (RCN) from Cα coordinates.

    Graph representation
    --------------------
    Nodes : one per residue
        node_labels[i] → physicochemical string (h=0 WL feature)
    Edges : pairs (i, j) with Cα–Cα distance < CONTACT_THRESHOLD_A
        adj[i] → list of (j, distance_bin_str) tuples

    The distance_bin string on each edge is what makes this an
    *edge-conditioned* WL kernel.  In the aggregation step below, each
    neighbour's contribution is labelled not just by its own node label but
    also by the type of contact (close / medium / far), allowing the kernel
    to discriminate between e.g. tight hydrophobic cores and loose surface
    contacts even when the participating residue types are identical.

    Parameters
    ----------
    pdb_data : dict
        Output of fetch_and_parse_pdb (keys: 'residues', 'ca_coords').

    Returns
    -------
    dict with keys:
        'node_labels'   : list[str]          — initial physicochemical labels
        'adj'           : list[list[tuple]]  — adjacency: adj[i] = [(j, bin), …]
        'n_nodes'       : int
    """
    residues  = pdb_data["residues"]
    coords    = pdb_data["ca_coords"]
    n         = len(residues)

    # Initialise h=0 node labels from the physicochemical property map
    node_labels = [AA_PHYSCHEM.get(res, AA_PHYSCHEM["UNK"]) for res in residues]

    # Adjacency list: adj[i] = list of (j, distance_bin_string)
    adj: List[List[Tuple[int, str]]] = [[] for _ in range(n)]

    # Compute pairwise Cα–Cα distance matrix (vectorised, O(N²) but fast)
    # coords shape: (N, 3)
    diff   = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist_m = np.linalg.norm(diff, axis=-1)             # (N, N)  Euclidean distances

    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_m[i, j])
            if d < CONTACT_THRESHOLD_A:
                bin_label = get_distance_bin(d)
                # Undirected edge: store in both directions
                adj[i].append((j, bin_label))
                adj[j].append((i, bin_label))

    return {"node_labels": node_labels, "adj": adj, "n_nodes": n}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CUSTOM EDGE-CONDITIONED WL GRAPH KERNEL  (from scratch, no grakel)
# ══════════════════════════════════════════════════════════════════════════════

class EdgeConditionedWLKernel:
    """
    Edge-Conditioned Weisfeiler-Lehman (WL) Graph Kernel — pure Python/NumPy.

    How the standard WL kernel works
    ---------------------------------
    In the original WL subtree kernel (Shervashidze et al., 2011), at each
    iteration h every node aggregates its neighbours' labels, sorts them,
    concatenates everything into a string, and hashes that string into a new
    compact integer label.  After H iterations the kernel is the dot product
    of the label-count histograms.

    Edge-conditioning extension (this implementation)
    --------------------------------------------------
    Instead of concatenating only the neighbour's label, we concatenate the
    neighbour's label **and the edge distance bin** before sorting.  This
    means two nodes in structurally identical local environments but with
    different contact geometries (e.g. one surrounded by close hydrophobic
    contacts vs. one surrounded by far hydrophobic contacts) will receive
    *different* hashed labels.  The kernel therefore captures not just
    topology but also the geometric intimacy of each structural motif.

    Parameters
    ----------
    n_iter : int
        Number of WL refinement iterations H.  Higher → captures larger
        neighbourhoods.  Typical range: 2–5.

    Attributes
    ----------
    label_dict : dict[str → int]
        Global hash table mapping every unique aggregated string encountered
        across all graphs and all iterations to a canonical integer index.
        This is the "structural motif dictionary" — each new entry represents
        a previously unseen neighbourhood pattern.
    next_label  : int
        Monotonically increasing counter for assigning fresh integer labels.
    """

    def __init__(self, n_iter: int = 3):
        self.n_iter    = n_iter
        # ── Global label→integer hash table ──────────────────────────────────
        # Every unique neighbourhood string (node label + sorted edge-conditioned
        # neighbour tokens) maps to a single integer.  Once assigned, the integer
        # never changes, so histograms across different graphs are comparable.
        self.label_dict: Dict[str, int] = {}
        self.next_label: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_create_label(self, s: str) -> int:
        """
        Look up the integer label for string s.
        If s has never been seen before, register it in label_dict.

        This is the hashing step that compresses variable-length
        neighbourhood strings into compact integers for histogram counting.
        Every new entry in label_dict corresponds to a newly discovered
        structural motif in the protein contact network.
        """
        if s not in self.label_dict:
            # New motif found — assign the next available integer.
            self.label_dict[s] = self.next_label
            self.next_label += 1
        return self.label_dict[s]

    def _relabel_graph(
        self,
        node_labels: List[str],
        adj: List[List[Tuple[int, str]]],
    ) -> List[str]:
        """
        Perform one WL iteration: compute h+1 labels from h labels.

        For each node i
        ---------------
        1. Start with its own current label L_i.
        2. For each neighbour j connected via edge with bin b:
               token = L_j + "|" + b      ← edge-conditioning step
           This is the key innovation: the edge bin becomes part of the
           neighbour's identity, so "close Hydrophobic" ≠ "far Hydrophobic".
        3. Sort all neighbour tokens lexicographically (order-invariant
           aggregation, mirrors the permutation-invariant set aggregation
           in the WL 1 test of graph isomorphism).
        4. Concatenate: new_string = L_i + "#" + ",".join(sorted_tokens)
        5. Hash new_string → integer via label_dict.

        Returns
        -------
        list[str] — new node labels for the next iteration (stored as strings
        of their integer indices so downstream concatenation still works).
        """
        new_labels: List[str] = []

        for i, own_label in enumerate(node_labels):
            # ── Build edge-conditioned neighbour tokens ───────────────────
            # Each token = neighbour_label + "|" + distance_bin.
            # The pipe-separated format ensures "closeHydrophobic" cannot
            # spuriously equal "clos" + "eHydrophobic" (no accidental merge).
            neighbour_tokens = []
            for j, dist_bin in adj[i]:
                # dist_bin ∈ {"close", "medium", "far"}
                token = node_labels[j] + "|" + dist_bin
                neighbour_tokens.append(token)

            # ── Sort tokens — makes aggregation permutation-invariant ─────
            # If we did NOT sort, two nodes with the same neighbour multiset
            # in different orders would get different labels (wrong).
            neighbour_tokens.sort()

            # ── Build the full aggregation string for hashing ─────────────
            aggregated = own_label + "#" + ",".join(neighbour_tokens)

            # ── Hash to a canonical integer label ─────────────────────────
            # _get_or_create_label updates label_dict when a new motif is found.
            new_int_label = self._get_or_create_label(aggregated)
            new_labels.append(str(new_int_label))

        return new_labels

    def _compute_histogram(
        self,
        all_iter_labels: List[List[str]],
    ) -> Dict[int, int]:
        """
        Aggregate label counts from all WL iterations into one histogram.

        The histogram h_G maps integer_label → count.
        Iterating over ALL iterations (not just the last) means the kernel
        captures both local (h=0,1) and more global (h=2,3) structural motifs,
        which is essential for discriminating proteins that share local motifs
        but differ in their higher-order fold geometry.
        """
        hist: Dict[int, int] = defaultdict(int)
        for labels_at_iter_h in all_iter_labels:
            for lbl in labels_at_iter_h:
                int_lbl = self._get_or_create_label(lbl)
                hist[int_lbl] += 1
        return dict(hist)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, graphs: List[dict]) -> np.ndarray:
        """
        Compute the N×N WL kernel matrix from a list of RCN graph dicts.

        Each graph dict must have:
            'node_labels' : list[str]          — h=0 physicochemical labels
            'adj'         : list[list[tuple]]  — adjacency (edge-conditioned)
            'n_nodes'     : int

        Algorithm
        ---------
        1. For each graph, run n_iter WL refinement steps, collecting all
           iteration label lists.
        2. Build per-graph histograms from all collected labels.
        3. Convert histograms to sparse vectors aligned to a shared vocabulary
           (label_dict).
        4. Return K = X @ X.T  (dot-product kernel).

        Returns
        -------
        K : np.ndarray of shape (N, N)
            Symmetric positive-semi-definite kernel matrix.
        """
        # ── Step 1: WL label propagation for every graph ──────────────────
        all_histograms: List[Dict[int, int]] = []

        for g_idx, graph in enumerate(tqdm(graphs, desc="WL refinement")):
            node_labels: List[str] = list(graph["node_labels"])   # copy
            adj         = graph["adj"]

            # Store labels at every iteration (including h=0)
            all_iter_labels: List[List[str]] = [list(node_labels)]

            for h in range(self.n_iter):
                # Perform one edge-conditioned WL refinement step
                node_labels = self._relabel_graph(node_labels, adj)
                all_iter_labels.append(list(node_labels))

            # Build histogram for this graph over all iterations
            hist = self._compute_histogram(all_iter_labels)
            all_histograms.append(hist)

        # ── Step 2: Align histograms to a shared integer vocabulary ───────
        vocab_size = self.next_label   # total unique labels discovered
        n_graphs   = len(all_histograms)

        # Feature matrix X: shape (n_graphs, vocab_size)
        X = np.zeros((n_graphs, vocab_size), dtype=np.float64)
        for i, hist in enumerate(all_histograms):
            for label_int, count in hist.items():
                if label_int < vocab_size:
                    X[i, label_int] = count

        # ── Step 3: Compute dot-product kernel matrix K = X @ X.T ────────
        # This is the standard inner product (linear kernel) in histogram space.
        # K[i,j] counts the total number of matching labelled subtrees between
        # graph i and graph j across all WL iterations — a structural similarity
        # score biologically interpretable as "shared structural motif frequency".
        K = X @ X.T

        # Cache feature matrix for potential test-time use
        self._X_train = X

        print(f"\n[WL Kernel] vocab size = {vocab_size} unique structural motifs")
        print(f"[WL Kernel] kernel matrix shape: {K.shape}")
        return K

    def transform(self, graphs: List[dict]) -> np.ndarray:
        """
        Compute kernel values between new (test) graphs and training graphs.
        Must be called after fit_transform.

        Returns
        -------
        K_test : np.ndarray of shape (n_test, n_train)
        """
        if not hasattr(self, "_X_train"):
            raise RuntimeError("Call fit_transform first.")

        vocab_size = self._X_train.shape[1]
        all_histograms: List[Dict[int, int]] = []

        for graph in tqdm(graphs, desc="WL transform (test)"):
            node_labels: List[str] = list(graph["node_labels"])
            adj         = graph["adj"]
            all_iter_labels: List[List[str]] = [list(node_labels)]

            for h in range(self.n_iter):
                node_labels = self._relabel_graph(node_labels, adj)
                all_iter_labels.append(list(node_labels))

            hist = self._compute_histogram(all_iter_labels)
            all_histograms.append(hist)

        X_test = np.zeros((len(all_histograms), vocab_size), dtype=np.float64)
        for i, hist in enumerate(all_histograms):
            for label_int, count in hist.items():
                if label_int < vocab_size:   # ignore unseen labels at test time
                    X_test[i, label_int] = count

        # K_test[i,j] = inner product of test graph i with train graph j
        return X_test @ self._X_train.T


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FULL PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    csv_path: str = "fireprotdb_20251015-164116.csv",
    max_proteins: int = 50,
    wl_iters: int = 3,
    allow_synthetic: bool = False,
):
    """
    End-to-end pipeline:
      Real FireProtDB CSV → PDB fetch → RCN build → WL kernel → SVM → report

    Parameters
    ----------
    csv_path        : path to the FireProtDB CSV export.
    max_proteins    : how many unique proteins to use (keeps kernel tractable).
                      Set to None to use all 175 available proteins.
    wl_iters        : number of WL refinement iterations.
    allow_synthetic : fall back to synthetic Cα data if RCSB download fails.
                      Default False for real-data runs; set True for offline demos.
    """
    print("=" * 70)
    print("  PROTEIN DEGRADATION PREDICTION — Custom Edge-Conditioned WL Kernel")
    print("=" * 70)

    # ── 1. Load real FireProtDB CSV ───────────────────────────────────────
    print(f"\n[Step 1] Loading real FireProtDB data from '{csv_path}' …")
    df = load_fireprotdb(csv_path, max_proteins=max_proteins)
    print("\n  Sample rows:")
    print(df.head(10).to_string(index=False))

    # ── 2. Fetch PDB structures & build RCNs ──────────────────────────────
    print("\n[Step 2] Fetching PDB structures and constructing RCNs …")

    pdb_cache: Dict[str, Optional[dict]] = {}   # avoid re-downloading
    graphs:    List[dict]  = []
    labels:    List[int]   = []
    meta:      List[dict]  = []

    for _, row in df.iterrows():
        pdb_id = row["pdb_id"]
        label  = int(row["label"])
        ddG    = float(row["ddG"])

        # Fetch real PDB from RCSB; fall back to synthetic only if enabled
        pdb_data = get_pdb_data(pdb_id, cache=pdb_cache,
                                allow_synthetic=allow_synthetic)
        if pdb_data is None:
            print(f"  [SKIP] {pdb_id} — could not build RCN.")
            continue

        # Build Residue Contact Network
        rcn = build_rcn(pdb_data)

        # Count edges for diagnostics
        n_edges = sum(len(neighbours) for neighbours in rcn["adj"]) // 2
        print(f"  [RCN] {pdb_id:6s} | nodes={rcn['n_nodes']:4d} | "
              f"edges={n_edges:5d} | ΔΔG={ddG:+.2f} | label={label}")

        graphs.append(rcn)
        labels.append(label)
        meta.append({"pdb_id": pdb_id, "ddG": ddG})

    if len(graphs) < 4:
        print("[ERROR] Too few graphs built — need at least 4 for train/test split.")
        sys.exit(1)

    labels = np.array(labels, dtype=int)
    print(f"\n  Total graphs: {len(graphs)}  |  "
          f"Stable={labels.sum()}  |  Unstable={(labels==0).sum()}")

    # ── 3. Custom Edge-Conditioned WL Kernel ──────────────────────────────
    print(f"\n[Step 3] Computing custom Edge-Conditioned WL kernel (iters={wl_iters}) …")
    wl_kernel = EdgeConditionedWLKernel(n_iter=wl_iters)
    K_full    = wl_kernel.fit_transform(graphs)   # shape (N, N)

    # Normalise kernel matrix: K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j])
    # This maps the kernel values to [-1, 1] (analogous to cosine similarity),
    # preventing graphs with more nodes from dominating purely due to size.
    diag      = np.sqrt(np.diag(K_full))
    diag_safe = np.where(diag == 0, 1.0, diag)   # avoid division by zero
    K_norm    = K_full / np.outer(diag_safe, diag_safe)
    K_norm    = np.clip(K_norm, 0.0, None)         # ensure non-negativity (PSD)

    # Only print the full matrix for small datasets (≤20 proteins)
    if len(graphs) <= 20:
        print("\nNormalised Kernel Matrix (K_norm):")
        print(np.round(K_norm, 3))
    else:
        print(f"\nKernel matrix computed: {K_norm.shape}  "
              f"min={K_norm.min():.3f}  max={K_norm.max():.3f}  "
              f"mean={K_norm.mean():.3f}")

    # ── 4. Train/Test split on the precomputed kernel matrix ──────────────
    print("\n[Step 4] Splitting kernel matrix into train / test sets …")
    n_samples = len(graphs)
    indices   = np.arange(n_samples)

    # Stratified split preserves class balance in both splits
    idx_train, idx_test = train_test_split(
        indices,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    # For a precomputed kernel SVM:
    #   K_train[i, j] = K(train_i, train_j)  → shape (n_train, n_train)
    #   K_test [i, j] = K(test_i,  train_j)  → shape (n_test,  n_train)
    K_train = K_norm[np.ix_(idx_train, idx_train)]
    K_test  = K_norm[np.ix_(idx_test,  idx_train)]
    y_train = labels[idx_train]
    y_test  = labels[idx_test]

    print(f"  Train: {len(idx_train)} samples  |  Test: {len(idx_test)} samples")
    print(f"  Train class dist: Stable={y_train.sum()} | "
          f"Unstable={(y_train==0).sum()}")
    print(f"  Test  class dist: Stable={y_test.sum()}  | "
          f"Unstable={(y_test==0).sum()}")

    # ── 5. SVM with precomputed kernel ────────────────────────────────────
    print("\n[Step 5] Training SVC (precomputed WL kernel) …")
    svc = SVC(
        kernel="precomputed",
        class_weight="balanced",   # compensate for 70/30 class imbalance
        C=1.0,
        random_state=42,
    )
    svc.fit(K_train, y_train)

    y_pred = svc.predict(K_test)

    # ── 6. Evaluation ─────────────────────────────────────────────────────
    print("\n[Step 6] Classification Report:")
    print("-" * 50)
    report = classification_report(
        y_test, y_pred,
        target_names=["Unstable (ΔΔG<0)", "Stable (ΔΔG≥0)"],
        zero_division=0,
    )
    print(report)

    # ── 7. Motif analysis ─────────────────────────────────────────────────
    print("[Step 7] Top structural motifs discovered by WL kernel:")
    print("-" * 50)
    # Sort by label integer (order of discovery = BFS-like expansion order)
    sorted_motifs = sorted(wl_kernel.label_dict.items(), key=lambda x: x[1])
    for motif_str, motif_id in sorted_motifs[:20]:
        display = (motif_str[:80] + "…") if len(motif_str) > 80 else motif_str
        print(f"  [{motif_id:4d}] {display}")
    if len(sorted_motifs) > 20:
        print(f"  … and {len(sorted_motifs) - 20} more motifs.")

    print("\n[Done] Pipeline completed successfully.")
    return {
        "kernel_matrix": K_norm,
        "wl_kernel_obj": wl_kernel,
        "graphs": graphs,
        "labels": labels,
        "svc": svc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_pipeline(
        csv_path     = "fireprotdb_20251015-164116.csv",
        max_proteins = 50,     # use 50 unique proteins for a fast first run
        wl_iters     = 3,      # WL refinement depth
        allow_synthetic = False,  # require real PDB downloads from RCSB
    )

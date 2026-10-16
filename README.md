# Protein Stability Prediction with an Edge-Conditioned WL Graph Kernel

This project is a graph machine-learning capstone for predicting whether a protein is broadly **stabilizing** or **destabilizing** according to its reported mutation stability measurements. It represents each protein structure as a **residue contact network (RCN)** and compares those networks with a custom, from-scratch **edge-conditioned Weisfeiler-Lehman (WL) graph kernel**.

The implementation is intentionally transparent: it uses Python, NumPy, Biopython, pandas, and scikit-learn, but does not use a graph-kernel library such as GraKeL. The WL feature construction, edge labels, histogram representation, and precomputed-kernel SVM workflow are implemented in `protein_degradation_pipeline.py`.

## Project question

Can coarse structural patterns in a protein's 3-D residue contact network help distinguish proteins with non-negative versus negative median mutation stability values?

The project uses FireProtDB measurements as a source of protein stability data:

- `DDG` is the mutation stability change in kcal/mol.
- `WWPDB` contains one or more PDB accessions for the wild-type structure.
- A protein is labeled **stable** when its aggregated `DDG >= 0`.
- A protein is labeled **unstable** when its aggregated `DDG < 0`.

This is a protein-level classification experiment, not a mutation-level predictor. Multiple mutations associated with the same PDB structure are reduced to one median `DDG` before the graph model is trained.

## Method overview

The complete workflow is:

```text
FireProtDB CSV
      |
      v
Filter DDG + WWPDB, parse PDB IDs, aggregate by PDB
      |
      v
Fetch or read PDB structure and extract C-alpha atoms
      |
      v
Build residue contact network
      |
      v
Edge-conditioned WL refinement and motif histograms
      |
      v
Normalized precomputed kernel matrix
      |
      v
SVC classifier and classification report
```

### 1. FireProtDB preparation

`load_fireprotdb()` reads only the `DDG` and `WWPDB` columns. This matters because a full FireProtDB export can be very large. Rows without both fields are removed, and the first valid four-character PDB accession is extracted from values such as `1YGW, 1RN1`.

All records for one PDB ID are grouped together and assigned the median `DDG`. This prevents the same structural graph from appearing repeatedly with different mutation rows. The default run then samples at most 50 proteins while preserving the stable/unstable class balance as far as possible.

### 2. Protein structures and residue contact networks

For each PDB ID, `get_pdb_data()` first looks in `pdb_files/`, then tries to download the structure from RCSB, and finally uses a Biopython fallback downloader. The parser keeps the first model and standard amino-acid residues that contain a C-alpha atom.

Each protein becomes an undirected graph:

- **Node:** one amino-acid residue.
- **Node label:** a coarse physicochemical category combining hydrophobicity and charge, such as `Hydrophobic_Neutral` or `Hydrophilic_Positive`.
- **Edge:** two residues whose C-alpha atoms are less than 8 Angstrom apart.
- **Edge label:** the C-alpha distance bin: `close` for `< 4`, `medium` for `4-6`, or `far` for `6-8` Angstrom.

The edge label adds a small amount of geometric information to an otherwise unweighted contact graph. The graph is deliberately a compact abstraction rather than a full molecular simulation.

### 3. Edge-conditioned Weisfeiler-Lehman refinement

The standard WL subtree procedure repeatedly relabels every node using its current label and the multiset of neighboring labels. Here, each neighbor token also includes the edge-distance bin:

```text
neighbor token = neighbor label + "|" + distance bin
new label      = current label + "#" + sorted neighbor tokens
```

Sorting makes aggregation invariant to the order in which neighbors are stored. The resulting string is mapped to a shared integer vocabulary. For every graph, the implementation retains labels from the initial state `h=0` and from every refinement iteration through `h=n_iter`.

For graph $G$, let $\phi(G)$ be the histogram of all labels observed across those iterations. The raw WL kernel is the dot product

$$
K(G_i, G_j) = \langle \phi(G_i), \phi(G_j) \rangle.
$$

The pipeline then normalizes the matrix by its diagonal:

$$
K_{norm}(i,j) = \frac{K(i,j)}{\sqrt{K(i,i)K(j,j)}}.
$$

This reduces the tendency for larger proteins to receive larger raw similarity scores merely because they contain more nodes and motifs.

### 4. Classification

The normalized matrix is passed to scikit-learn's `SVC` with `kernel="precomputed"`. For the train/test split:

- `K_train` contains train-to-train similarities.
- `K_test` contains test-to-train similarities.
- The split is 70/30 and stratified with `random_state=42`.
- `class_weight="balanced"` compensates for class imbalance.

The script prints a classification report with precision, recall, F1 score, and support for both classes. It also prints the first 20 structural motifs discovered by the WL label dictionary.

## Repository layout

```text
.
├── protein_degradation_pipeline.py  # End-to-end data, graph, kernel, and SVM pipeline
├── pdb_files/                        # Local PDB cache and bundled structures
└── README.md                         # Project documentation
```

The FireProtDB CSV is an external input expected at the project root by the current entry point. The default filename is `fireprotdb_20251015-164116.csv`; the file may be absent from a fresh clone because large data exports are ignored by Git and must be obtained separately.

## Requirements

- Python 3.9 or newer
- Internet access for PDBs that are not already present locally, unless synthetic fallback is enabled
- Enough memory for the FireProtDB CSV and the pairwise graph/kernel computations

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn biopython tqdm
```

## Running the pipeline

Run the default experiment from the repository root:

```bash
python protein_degradation_pipeline.py
```

The default entry point uses:

- `fireprotdb_20251015-164116.csv`
- at most 50 proteins
- 3 WL refinement iterations
- real PDB structures only (`allow_synthetic=False`)

The program reports progress for each stage, including the number of usable FireProtDB records, downloaded or cached structures, graph sizes, kernel dimensions, train/test class counts, and the final classification report.

### Offline/demo execution

The code contains a synthetic C-alpha generator for demos when RCSB cannot be reached. It is intentionally disabled in the production-style default because synthetic structures should not be mixed silently with real FireProtDB labels.

To use it, call the function from a Python session or a small wrapper:

```python
from protein_degradation_pipeline import run_pipeline

results = run_pipeline(
    csv_path="fireprotdb_20251015-164116.csv",
    max_proteins=20,
    wl_iters=3,
    allow_synthetic=True,
)
```

The returned dictionary contains the normalized kernel matrix, the fitted WL kernel object, graph data, labels, and the trained SVC.

## Important implementation details

### WL vocabulary and test-time behavior

`fit_transform()` creates the shared motif vocabulary from the training call's graph list and stores the corresponding feature matrix. `transform()` can then compute similarities for new graphs against that stored training matrix. Labels discovered only in a later graph are ignored if they are outside the fitted feature vocabulary, which is the expected behavior for a fixed feature space.

### Reproducibility

The FireProtDB subsample and train/test split use `random_state=42`. PDB downloads are cached in `pdb_files/`, so subsequent runs do not need to download structures that are already present. For a fully reproducible experiment, preserve the input CSV, the cached PDB files, the Python package versions, and the run configuration.

### Computational cost

The contact graph construction computes an $N \times N$ pairwise distance matrix per protein. The WL kernel creates an $M$-dimensional histogram space, where $M$ is the number of distinct labels discovered across all graphs and iterations, and computes `X @ X.T`. The resulting kernel matrix is $P \times P$ for $P$ proteins. Increasing `max_proteins`, protein size, or `wl_iters` increases runtime and memory use.

## Interpreting the results

The kernel score measures shared labeled local neighborhoods, not a causal physical stability mechanism. A useful capstone analysis should therefore compare:

- Different WL depths, such as 1, 2, 3, and 4 iterations.
- Different contact thresholds or distance bins.
- The edge-conditioned kernel against a node-only WL baseline.
- Balanced accuracy, macro F1, and confusion matrices in addition to accuracy.
- A single split against repeated or cross-validated splits.

Because the current pipeline uses one structure-level label derived from an aggregate of mutations, its result should be described as a structural classification experiment. It should not be interpreted as predicting the effect of a particular amino-acid substitution without adding mutation-position and mutation-identity features.

## Limitations and responsible use

- The binary target `DDG >= 0` is a modeling choice; it discards the magnitude and uncertainty of the original stability measurement.
- Median aggregation can hide important mutation-level variation within a PDB entry.
- PDB structures vary in resolution, missing residues, chain composition, and experimental conditions.
- The current parser uses the first model and combines standard residues across chains into one graph.
- Randomly splitting proteins can still leave dataset-level biases or related structures across train and test sets.
- Synthetic fallback graphs are suitable for smoke tests and demonstrations, not for scientific conclusions.
- A strong classification report on this small, capped dataset is not evidence of generalization to all proteins.

## Suggested capstone extensions

1. Replace the binary label with regression on median `DDG` or a three-class stability target.
2. Keep chain identifiers and compare single-chain graphs with multi-chain graphs.
3. Add residue-level features such as solvent exposure, secondary structure, or amino-acid identity.
4. Evaluate grouped cross-validation by protein family or sequence similarity.
5. Compare this custom kernel with a standard WL implementation as a correctness check.
6. Add kernel alignment, hyperparameter search, and confidence intervals over repeated splits.
7. Visualize the most frequent WL motifs and inspect their corresponding residues in representative structures.

## References

- N. Shervashidze et al., _Weisfeiler-Lehman Graph Kernels_, Journal of Machine Learning Research, 2011.
- FireProtDB: a database of experimentally measured protein stability changes.
- RCSB Protein Data Bank: https://www.rcsb.org/
- Biopython documentation: https://biopython.org/
- scikit-learn documentation for precomputed-kernel SVC: https://scikit-learn.org/

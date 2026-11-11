# **A Circle Stroke-Driven Method for Graph Construction of Oracle Bone Characters**

Reference implementation and datasets for the paper  
**"A Circle Stroke-Driven Method for Graph Construction of Oracle Bone Characters"**.

This repository contains the proposed method (method A), five re-implemented  
baselines (methods B–F), the ablation and parameter-sensitivity scripts, the  
input skeleton images, the ground-truth adjacency matrices, and the figures and  
tables reported in the paper.

> **Status:** code and data accompanying a manuscript under preparation.  
> If you find a discrepancy between the code and the paper, please open an issue.

---

## 1. Overview

The pipeline turns a binary skeleton image of an oracle bone character into an  
attributed graph:

1. **Endpoint detection** – branch-length verification on the skeleton.
2. **Junction detection** – skeleton pixels with more than two neighbours.
3. **Circular keypoint detection** – midpoint sampling along skeleton branches,  
   with local tracking and Chebyshev de-duplication.
4. **Type-sensitive node merging** – nearby keypoints of the same type are  
   merged, with a priority rule that protects junctions and endpoints.
5. **Global midpoint insertion** and **redundant corner removal**.
6. **Degree-limited edge building** – edges are added under a node-degree cap,  
   producing the final adjacency matrix.

Each of these stages can be switched off through the `USE_*` flags at the top of  
`code/method_A/src/main.py`, which is how the ablation settings are produced.

---

## 2. Repository layout

```
.
├── code/
│   ├── paths.py                      # central, relative path configuration
│   ├── method_A/src/                 # proposed method
│   │   ├── main.py                   # entry point
│   │   ├── endpoint_detector.py
│   │   ├── junction_detector.py
│   │   ├── midpoint_detector.py
│   │   ├── global_midpoint_detector.py
│   │   ├── corner_remover.py
│   │   ├── degree_edge_builder.py
│   │   ├── graph_utils.py
│   │   ├── graph_postprocessor.py
│   │   └── visualizer.py
│   ├── baselines/
│   │   ├── method_B/                 # Zhang et al., glyph graph isomorphism network
│   │   ├── method_C/                 # Arrivault et al., fuzzy hierarchical attributed graph
│   │   ├── method_D/                 # Tang et al., graph feature extraction
│   │   ├── method_E/                 # Stauffer et al., graph-based keyword spotting
│   │   └── method_F/                 # Sun et al., graph-matching seal characters
│   └── compare_code/
│       ├── ablation_study.py                    # ablation evaluation against ground truth
│       ├── Parameter_sensitivity_analysis.py    # parameter sweep
│       └── Parameter_sensitivity_analysis_compare.py  # sweep aggregation
├── data/
│   ├── skeleton_756/  skeleton_1512/  skeleton_3780/                  # input skeletons
│   ├── Matrix_756/    Matrix_1512/    Matrix_3780/                     # ground-truth matrices
│   └── Image_3780/                                                     # original character images
├── figure/                           # figures 1–5 of the paper
├── table/                            # tables 1–5 of the paper
├── requirements.txt                  # dependencies for the whole repository
├── NOTICE.md                         # third-party data attribution
└── LICENSE
```

`Experiment_results/` is **not** tracked in git (see `.gitignore`); it is created  
automatically the first time any script runs.

---

## 3. Environment

Tested with **Python 3.13** on Windows. Linux and macOS should work unchanged —  
no absolute paths are hard-coded anywhere.

```bash
git clone <this-repo>.git
cd <this-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Additional dependencies are declared per sub-project:

- `code/method_A/requirements.txt` – proposed method
- `code/baselines/requirements.txt` – baselines B–F

---

## 4. Data

Three dataset scales are provided. `skeleton_N` holds the input skeleton images  
and `Matrix_N` the ground-truth adjacency matrices (`*.csv`); the two always  
share the same file stems.

| Scale | Input skeletons      | Ground-truth matrices | Notes                       |
| ----- | -------------------- | --------------------- | --------------------------- |
| 756   | `data/skeleton_756`  | `data/Matrix_756`     | 756 characters              |
| 1512  | `data/skeleton_1512` | `data/Matrix_1512`    | 1512 characters             |
| 3780  | `data/skeleton_3780` | `data/Matrix_3780`    | full set used in the paper  |

### Provenance

All 3,780 glyph images and their ground-truth graphs come from the
**Graph-Based Evolutionary Dataset for Oracle Bone Characters (GBEDOBC)**:

> Q. Jiao, J. Wu, Q. Liu, H. Zhang, Z. Zhang, B. Li, J. Xiong, G. Liu and Y. Liu,
> "A graph-based evolutionary dataset for oracle bone characters from
> inscriptions to modern Chinese scripts," *npj Heritage Science*, vol. 13,
> no. 1, Art. no. 369, 2025.
> DOI: [10.1038/s40494-025-01951-0](https://doi.org/10.1038/s40494-025-01951-0)

- Dataset repository: <https://github.com/BrisksHan/GBEDOBC>
- The glyph images in GBEDOBC are themselves taken from Wang et al.,
  *PLoS ONE* 17, e0272974 (2022),
  DOI: [10.1371/journal.pone.0272974](https://doi.org/10.1371/journal.pone.0272974)

**If you use `data/`, you must cite the GBEDOBC paper in addition to ours** —
see [NOTICE.md](NOTICE.md) for the full provenance chain and BibTeX entries.

---

## 5. Usage

All scripts resolve their paths relative to the repository root through  
`code/paths.py`, so they can be launched from any working directory.

### 5.1 Proposed method

```bash
python code/method_A/src/main.py
```

| Setting       | Where                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------ |
| Dataset       | `DATASET` at the top of `main.py` (`"skeleton_756"`, `"skeleton_1512"`, `"skeleton_3780"`) |
| Stages to run | the `USE_*` flags inside `main()`                                                          |

Output:

```
Experiment_results/A_matrix_<N>/              # <stem>_adj.npy, <stem>_nodes.npy
Experiment_results/A_matrix_<N>_visualization/
```

### 5.2 Global configuration

`code/paths.py` is the single source of truth for paths. The default dataset is  
set by `DEFAULT_DATASET`; the environment variables below override the  
directories without touching any code:

| Variable              | Meaning                             |
| --------------------- | ----------------------------------- |
| `SKELETON_DATASET`    | dataset to use, e.g. `skeleton_756` |
| `SKELETON_DATA_DIR`   | override `data/`                    |
| `SKELETON_INPUT_DIR`  | override the skeleton directory     |
| `SKELETON_GT_DIR`     | override the ground-truth directory |
| `SKELETON_RESULT_DIR` | override `Experiment_results/`      |

To check the configuration that is in effect:

```bash
python code/paths.py
```

### 5.3 Baselines

```bash
python code/baselines/method_B/main_B.py
python code/baselines/method_C/main_Egy.py
python code/baselines/method_D/main_D.py
python code/baselines/method_E/main_m.py
python code/baselines/method_F/main_F.py
```

Each writes to `Experiment_results/baseline/<M>_matrix_<N>/` and  
`..._visualization/`. The dataset is selected by the `DATASET` line at the top of  
each `main_*.py` (`DEFAULT_DATASET` means "follow `paths.py`").

### 5.4 Ablation

The five ablation settings reported in the paper correspond to the result  
directories suffixed `_t1` … `_t5`. Each setting is produced by toggling the  
`USE_*` flags in `code/method_A/src/main.py` and naming the output directory  
accordingly, e.g.:

```
Experiment_results/A_matrix_3780_t4/
```

`ablation_study.py` then compares one of these variants against the ground truth:

```bash
# edit a_dir inside the script to point at the variant you want to evaluate
python code/compare_code/ablation_study.py
```

It reports, averaged over all matched samples, nine structural agreement metrics  
against `data/Matrix_<N>/`:

| Metric                                        | Key            | Better      |
| --------------------------------------------- | -------------- | ----------- |
| node count                                    | `node_count_A` | close to GT |
| degree-distribution Jensen–Shannon divergence | `js_dist`      | ↓           |
| edge-density error                            | `dens_err`     | ↓           |
| degree-assortativity absolute error           | `assort_err`   | ↓           |
| cyclomatic-number absolute error              | `cyclo_err`    | ↓           |
| cycle-length-distribution JS divergence       | `cycle_js`     | ↓           |
| connected-component-count error               | `comp_err`     | ↓           |
| algebraic-connectivity error                  | `alg_conn_err` | ↓           |
| spectral distance                             | `spec_dist`    | ↓           |

Note that `ablation_study.py` evaluates a single variant directory, `_t4` by  
default; change `a_dir` inside the script to evaluate another setting.

### 5.5 Parameter sensitivity

```bash
python code/compare_code/Parameter_sensitivity_analysis.py          # full sweep
python code/compare_code/Parameter_sensitivity_analysis_compare.py  # aggregate
```

The sweep writes one directory per parameter value, e.g.  
`Experiment_results/parameter_sensitivity_analysis/A_matrix_3780_t5_merge_dist=4/`.  
This produces a large number of small files (tens of thousands per setting);  
budget disk space accordingly.

---

## 6. Figures and tables

- `figure/` – Figures 1–5. Figures 4 and 5 are generated by  
  `compare_code/Parameter_sensitivity_analysis_compare.py` from the sweep output;  
  Figures 1–3 are drawn manually and are included as static images.
- `table/` – Tables 1–5 as Excel workbooks.

---

## 7. Citation

If you use this code, please cite our paper (BibTeX to be added upon
publication) **and** the dataset publication that provides the images and
ground-truth graphs:

```bibtex
@article{jiao2025gbedobc,
  title   = {A graph-based evolutionary dataset for oracle bone characters
             from inscriptions to modern Chinese scripts},
  author  = {Jiao, Qingju and Wu, Jingwen and Liu, Qi and Zhang, Han and
             Zhang, Zhan and Li, Bang and Xiong, Jing and Liu, Guoying and
             Liu, Yongge},
  journal = {npj Heritage Science},
  volume  = {13},
  number  = {1},
  pages   = {369},
  year    = {2025},
  doi     = {10.1038/s40494-025-01951-0}
}
```

If you redistribute or publish results based on the glyph images themselves,
please also cite their original source:

```bibtex
@article{wang2022oracle,
  title   = {Study on the evolution of Chinese characters based on few-shot
             learning: From oracle bone inscriptions to regular script},
  author  = {Wang, M. and others},
  journal = {PLoS ONE},
  volume  = {17},
  pages   = {e0272974},
  year    = {2022},
  doi     = {10.1371/journal.pone.0272974}
}
```

A machine-readable form is provided in [CITATION.cff](CITATION.cff), and the
full attribution statement is in [NOTICE.md](NOTICE.md).

---

## 8. License

The source code in this repository is released under the **MIT License** — see  
[LICENSE](LICENSE).

The dataset files under `data/` are **not** covered by the MIT License. They are  
redistributed from a third-party dataset and remain subject to its terms; see  
[NOTICE.md](NOTICE.md) before reusing them.

---

## 9. Contact

Questions about the code or the paper: **2503011022@stu.aynu.edu.cn**

Issues and pull requests are welcome via the
[issue tracker](https://github.com/Renenjie/obc-skeleton-graph/issues).

# Business Entity Resolution — AWS ML Challenge 2026

End-to-end ML pipeline for the Business Entity Resolution Challenge. Given business records from 3 independent data sources (with noisy names and addresses), this pipeline determines which records refer to the same real-world business entity and outputs the required `matching_results.tsv` and `candidate_pairs.tsv` files.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset Setup](#dataset-setup)
3. [Environment Setup](#environment-setup)
4. [Running the Pipeline](#running-the-pipeline)
5. [Validation](#validation)
6. [Pipeline Architecture](#pipeline-architecture)
7. [Output Format](#output-format)

---

## Project Structure

```
aws_ml_challenge_2026/
│
├── dataset/                        ← Place downloaded dataset here (gitignored)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── output/                         ← Generated outputs land here (gitignored)
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── utils/
│   └── validate_submission.py      ← Official format validator (stdlib only)
│
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt
│       ├── README.md
│       └── src/
│           ├── main.py             ← Pipeline entry point
│           ├── data_io.py          ← Data loading & TSV saving
│           ├── blocking.py         ← TF-IDF + k-NN candidate generation
│           ├── features.py         ← String similarity feature engineering
│           └── model.py            ← XGBoost training, threshold tuning & inference
│
├── Documentation_template.md       ← Methodology write-up
└── .gitignore
```

---

## Dataset Setup

> ⚠️ **The `dataset/` folder is gitignored — never commit your data files.**

After downloading the dataset from the challenge portal:

1. Create the required folder structure at the **root of this repository**:
   ```
   dataset/
   ├── train/
   └── test/
   ```

2. Place the files exactly like this:

   | File | Location |
   |------|----------|
   | `train_source1.tsv` | `dataset/train/train_source1.tsv` |
   | `train_source2.tsv` | `dataset/train/train_source2.tsv` |
   | `train_source3.tsv` | `dataset/train/train_source3.tsv` |
   | `train_ground_truth.tsv` | `dataset/train/train_ground_truth.tsv` |
   | `test_source1.tsv` | `dataset/test/test_source1.tsv` |
   | `test_source2.tsv` | `dataset/test/test_source2.tsv` |
   | `test_source3.tsv` | `dataset/test/test_source3.tsv` |

3. If the `utils/` folder with `validate_submission.py` came with the challenge download, place it at the root of this repository too.

---

## Environment Setup

It is strongly recommended to use a Python virtual environment.

### Step 1 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Running the Pipeline

Run the full pipeline from the **root of the repository** with a single command:

**Windows (PowerShell):**
```powershell
python code/business_entity_resolution/src/main.py `
    --data_dir dataset `
    --output_dir output
```

**macOS / Linux:**
```bash
python3 code/business_entity_resolution/src/main.py \
    --data_dir dataset \
    --output_dir output
```

### What happens when you run it

The pipeline runs automatically in the following stages:

| Stage | Description |
|-------|-------------|
| **1. Load Training Data** | Reads all 4 training TSVs into memory |
| **2. Blocking (Train)** | Runs TF-IDF + k-NN to generate candidate pairs from training data |
| **3. Feature Engineering (Train)** | Computes Levenshtein, Jaro-Winkler, Jaccard features for each candidate pair |
| **4. Model Training** | Trains XGBoost classifier; sweeps threshold to maximize F0.5 on a validation split |
| **5. Load Test Data** | Reads all 3 test TSVs |
| **6. Blocking (Test)** | Generates candidate pairs for the test set |
| **7. Feature Engineering (Test)** | Computes features for test candidate pairs |
| **8. Inference** | Applies trained model at the optimized threshold |
| **9. Save Outputs** | Writes `matching_results.tsv` and `candidate_pairs.tsv` to `output/` |

Expected runtime: **~5–15 minutes** depending on dataset size and your CPU.

---

## Validation

After the pipeline finishes, validate your output format using the official checker:

**Windows (PowerShell):**
```powershell
python utils/validate_submission.py `
    --matching output/matching_results.tsv `
    --candidate output/candidate_pairs.tsv `
    --test-dir dataset/test
```

**macOS / Linux:**
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

You should see `PASS` (exit 0). If there are issues, the validator will list them for you to fix.

---

## Pipeline Architecture

```
Source 1   ─┐
Source 2   ─┤──▶  Blocking (TF-IDF + k-NN, per country)  ──▶  Candidate Pairs
Source 3   ─┘
                                    │
                                    ▼
                     Feature Engineering (per candidate pair)
                     ┌──────────────────────────────────┐
                     │ • Levenshtein distance (name)     │
                     │ • Jaro-Winkler similarity (name)  │
                     │ • Jaccard similarity (name)       │
                     │ • Exact match flag (name)         │
                     │ • Levenshtein distance (address)  │
                     │ • Jaro-Winkler similarity (addr)  │
                     │ • Jaccard similarity (address)    │
                     │ • Exact match flag (address)      │
                     │ • Length difference (name/addr)   │
                     └──────────────────────────────────┘
                                    │
                                    ▼
                     XGBoost Classifier (binary: match / no-match)
                     + F0.5-optimized probability threshold
                                    │
                                    ▼
                     matching_results.tsv  +  candidate_pairs.tsv
```

### Why this design?

- **Country-partitioned blocking** respects geographic boundaries and avoids comparing unrelated records, which keeps precision high.
- **Character-level TF-IDF (n-grams 2–4)** is robust to typos, abbreviations, and transliterations — far better than word-level indexing for noisy business names.
- **Jaro-Winkler** handles prefix matches especially well (company names typically share the same root).
- **F0.5 threshold sweeping** directly optimizes the scoring metric, not just accuracy. Since F0.5 penalizes false positives twice as much as false negatives, we let the data tell us the right operating point.

---

## Output Format

| File | Purpose |
|------|---------|
| `output/matching_results.tsv` | Upload this to the leaderboard |
| `output/candidate_pairs.tsv` | Included in the final submission zip |

Both files are tab-separated with columns described in the challenge specification.

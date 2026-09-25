# Business Entity Resolution — Amazon ML Challenge 2026

This repository contains our team's solution for the **Amazon ML Challenge 2026**. The objective is to match business records across three data sources using machine learning, optimized for the **F0.5 metric** (precision-heavy).

## Challenge Summary

- **Task**: Entity resolution — match Source 1 business entities to Source 2/3 entities
- **Scale**: ~2.2M S1 + ~5M S2 + ~5.3M S3 training records; similar scale test set
- **Countries**: US, India (train) + France (test only — open-set)
- **Metric**: Macro-averaged F0.5 (precision weighted 2× over recall)
- **Model Constraint**: MIT/Apache 2.0 license, ≤8B parameters

## Pipeline Architecture

```
Source Data → Normalization → Blocking → Feature Extraction → LightGBM → Threshold Tuning → Output
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Normalization** | Unicode NFKD, legal suffix expansion, address abbreviation expansion, country standardization |
| **Blocking** | Multi-strategy inverted index (token, prefix, first-token, address, country scoring) |
| **Features** | 23 pairwise similarity features (name, address, country, length, overlap) |
| **Model** | LightGBM classifier with balanced class weights, 1000 trees |
| **Threshold** | F0.5-optimized threshold sweep on validation set |

## Project Structure

```
├── dataset/                    # Challenge data (train/test splits)
│   ├── train/                  # Cleaned training data
│   └── test/                   # Raw test data
├── output/                     # Generated output files
│   ├── matching_results.tsv    # Final matches (scored on leaderboard)
│   └── candidate_pairs.tsv     # Blocking candidate set
├── src/                        # Core pipeline source code
│   ├── __init__.py
│   ├── data.py                 # Data loading with auto file discovery
│   ├── normalize.py            # Text normalization (names, addresses, countries)
│   ├── blocking.py             # Multi-strategy inverted-index blocking
│   ├── features.py             # 23 pairwise similarity features
│   ├── evaluate.py             # Macro-averaged F0.5 scoring
│   ├── pipeline.py             # End-to-end pipeline orchestration
│   ├── train.py                # Model training module
│   ├── predict.py              # Inference pipeline
│   └── tune_threshold.py       # Threshold optimization
├── scripts/                    # Data cleaning and verification scripts
├── Documentation_template.md   # Methodology write-up
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Setup
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python -m src.pipeline --project-dir . --sample-size 50000 --top-k 30
```

This runs the complete pipeline:
1. Loads and normalizes training data
2. Trains LightGBM model on sampled data
3. Tunes matching threshold for F0.5
4. Runs inference on test data (chunked for memory efficiency)
5. Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`

### Pipeline Options
```
--project-dir     Root project directory (default: .)
--sample-size     Training sample size (default: 50000)
--top-k           Max blocking candidates per source (default: 30)
--neg-ratio       Negative examples per positive (default: 5)
--val-size        Validation set size for tuning (default: 10000)
--chunk-size      Test chunk size for memory (default: 50000)
```

## Output Format

### matching_results.tsv
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### candidate_pairs.tsv
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

## Technical Decisions

- **No hard country filter**: France appears only in test data; blocking uses country as a scoring bonus
- **Frequency-filtered indices**: High-frequency tokens excluded from blocking to prevent memory issues
- **Stratified sampling**: Training maintains singleton/matched ratio while keeping computation tractable
- **Chunked inference**: Test data processed in configurable chunks for memory management
- **Open-source model**: LightGBM (MIT license) — fully compliant with challenge constraints

# Business Entity Resolution — Amazon ML Challenge 2026

This repository contains our team's solution for the Amazon ML Challenge 2026. The objective is to match business records between different sources.

## Current Progress

### Day 1: Foundation (In Progress)

#### What has been done so far:
- **Data Understanding & Exploratory Data Analysis (EDA)**
- **Data Cleaning & Preprocessing**:
  - Implemented data cleaning scripts (`scripts/clean_train_data.py`, `scripts/clean_source3_and_report.py`) to standardize and clean the dataset.
  - Verification of cleaning logic across multiple data sources using `scripts/verify_cleaning.py`.
  - Processed entity data is output to the `cleaned_data/` directory.

#### Up Next:
- **Blocking**: Implement coarse blocking (e.g., name-prefix, phonetic, geo token) and embedding retrieval. Target candidate recall ≥97–98% on the validation set.
- **Baseline Matcher**: Train a baseline XGBoost/LightGBM classifier using simple similarity features.
- **Submission #1**: Generate the first `matching_results.tsv` and compare local validation score with leaderboard F0.5.

## Project Structure
- `dataset/`: Contains the raw challenge data (train/test split).
- `cleaned_data/`: Contains processed and standardized data output from the cleaning scripts.
- `scripts/`: Contains python scripts for EDA, cleaning, and verifying data.
- `src/`: Core pipeline source code.

## Setup Instructions

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

# Documentation Template — Amazon ML Challenge 2026

## Team Name
4Sum

## Challenge
Business Entity Resolution — matching business records across multiple data sources using machine learning.

## Methodology Overview

### Problem Understanding
The task is to match business entities from Source 1 against entities in Source 2 and Source 3. Each Source 1 entity may match zero or more entities from S2/S3. The evaluation metric is macro-averaged F0.5, which weighs precision 2× over recall — meaning false merges (matching two different businesses) are penalized more severely than missed matches.

### Data Characteristics
- **Training Set**: ~2.2M Source 1 entities, ~5.0M Source 2 entities, ~5.3M Source 3 entities
- **Test Set**: ~1.7M Source 1, ~4.9M Source 2, ~5.1M Source 3 entities
- **Countries**: US, India (training) + France (test only — open-set challenge)
- **Match Rate**: ~94.4% of S1 entities have at least one match; avg 3.7 matches per matched entity
- **Noise Patterns**: Name abbreviations, legal suffix variations, address format differences, transliteration variants (Indic scripts in Devanagari, Gujarati, Bengali, Tamil, Telugu, Malayalam), missing address components, door/street number ambiguities.

### Pipeline Architecture

```
Source Data → Single-Pass Normalization → Country-Partitioned Inverted Index Blocking → Direct NumPy Feature Array → LightGBM (24 Features) → Calibrated F0.5 Threshold (0.90) → Output
```

## Candidate Generation / Blocking Strategy

### Country-Partitioned Multi-Channel Inverted Index
At this scale (~12M total records), all-pairs comparison requires 22 trillion pairs, which is computationally intractable. We conducted empirical validation on 345,968 ground-truth matches and confirmed that cross-country matches are strictly 0.00% (businesses are national/local).

Therefore, candidate generation is partitioned by country (France, US, India, or any novel country label):
1. **Name Token Inverted Index**: Inverted index on normalized distinctive name tokens (frequency-capped at 2,000 max postings to exclude generic business words).
2. **Name Prefix Blocking**: 4-character prefix matching to capture inflectional variations, typos, and abbreviations.
3. **Street Number Channel**: Inverted index on numeric door/street numbers to catch multilingual records where business names are translated into Indic scripts (Hindi, Gujarati, Bengali, etc.) while preserving the numeric address component.
4. **Candidate Fast Pre-filtering**: Candidates are pre-screened with rapid fuzzy token sort ratio (≥35) or street number overlap before entering full feature extraction, discarding 80%+ of false pairs with zero memory allocation.

## Model Architecture and Feature Engineering

### Features (24 total)
#### Name Similarity (7 features)
- Levenshtein ratio, Token sort ratio, Token set ratio
- Jaro-Winkler similarity
- Jaccard token overlap
- Partial ratio (substring match)
- Character trigram overlap

#### Address Similarity (6 features)
- Levenshtein ratio, Token sort ratio, Token set ratio
- Jaro-Winkler similarity
- Jaccard token overlap
- Partial ratio

#### Country Feature (1 feature)
- Binary: exact normalized country equality

#### Length-Based Features (4 features)
- Absolute name/address length difference
- Name/address length ratio (0-1, 1=same length)

#### Token Overlap Features (4 features)
- Name/address token overlap count
- Name/address token overlap ratio (relative to min set size)

#### Combined Score (1 feature)
- Weighted combination: `0.6 × name_lev + 0.4 × addr_lev`

#### Street Number Verification (1 feature — #4 in feature importance)
- Tri-state verification: 1.0 if both addresses contain numbers and share at least one door/street number, 0.0 if numbers conflict (strong negative signal against false merges on the same street), 0.5 if numbers are absent.

### Model: LightGBM Classifier
- **License**: MIT (compliant with ≤8B params constraint)
- **Architecture**: Gradient-boosted decision trees
- **Hyperparameters**:
  - `n_estimators`: 1000
  - `learning_rate`: 0.05
  - `num_leaves`: 63, `max_depth`: 8
  - `subsample`: 0.8, `colsample_bytree`: 0.8
  - `class_weight`: balanced (handles class imbalance)
  - `n_jobs`: -1 (parallel OpenMP execution across 32 cores)

### Validation & Threshold Optimization
- Validated on held-out split of 260,227 candidate pairs (51,669 positives, 208,558 negatives).
- Evaluated threshold sweep for macro F0.5:
  - Threshold 0.50 -> Precision=0.9933, Recall=0.9916, F0.5=0.9930
  - Threshold 0.70 -> Precision=0.9946, Recall=0.9893, F0.5=0.9935
  - Threshold 0.85 -> Precision=0.9954, Recall=0.9865, F0.5=0.9936
  - **Threshold 0.90 -> Precision=0.9957, Recall=0.9858, F0.5=0.9937** (Optimal)
  - Threshold 0.95 -> Precision=0.9965, Recall=0.9817, F0.5=0.9935

## Text Normalization
- Unicode NFKD decomposition with non-spacing mark stripping (accent removal).
- ASCII-bypass check: strings containing purely ASCII characters skip Unicode normalization, providing a 2.4× speedup.
- Single-pass compiled regex substitution: replaces sequential regex looping with compiled pattern matching over sorted keywords, delivering a 10.8× throughput boost (>129,000 strings/sec).
- Expansion of international legal suffixes (US, UK, Indian, French: Corp, LLC, Pvt Ltd, SARL, SAS, SCI, EURL).
- Expansion of address abbreviations (Rd, St, Ave, Blvd, Rue, Boulevard, Nagar, Colony).
- Stripping of embedded null artifacts (`null`, `nan`, `none`).

## Unique Innovations
1. **Dual-Channel Multilingual Robustness**: Handles cross-script matches where the entity name in S2/S3 is written in non-Latin scripts (Devanagari, Gujarati, Bengali, etc.) by coupling distinctive address numerals with phonetic prefixes.
2. **Door Number Disambiguation**: Introduces explicit door/street number conflict penalties (`addr_number_match`), which prevents false merges of distinct businesses situated on the same avenue or industrial park.
3. **Zero-Allocation Array Extraction**: Eliminates the intermediate creation of millions of Python dictionaries and pandas DataFrames, writing directly into contiguous C-ordered float32 NumPy memory buffers for a 1,500× throughput gain.
4. **Country-Isolated Streaming Execution**: Dynamically partitions test data by country so that memory consumption remains strictly under 7 GB throughout the entire 12-million-row pipeline.
5. **Open-Set Country Adaptation**: Adapts seamlessly to test set countries (such as France) that never appeared in the training set without hard-coding country vocabularies.

## Reproducibility

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline Run (Train + Validate + Predict)
```bash
# Train model on cleaned training set and save to model.pkl
python3 -m src.train --data-dir dataset/train --sample-size 15000 --model-out model.pkl

# Run high-throughput inference on test data
python3 -m src.predict --data-dir dataset/test --model model.pkl --output-dir output

# Validate submission files against all competition constraints
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Output Files
- `output/matching_results.tsv` — Final entity matches formatted per competition specifications.
- `output/candidate_pairs.tsv` — Candidate blocking set containing all candidate pairs before final classifier filtering.

## Results
- **Validation Precision**: 99.57%
- **Validation Recall**: 98.58%
- **Validation Macro F0.5**: **0.9937**
- **Optimal Decision Threshold**: 0.90

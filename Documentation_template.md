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
- **Noise Patterns**: Name abbreviations, legal suffix variations, address format differences, transliteration variants, missing address components, landmark-based references

### Pipeline Architecture

```
Source Data → Normalization → Blocking (Candidate Generation) → Feature Extraction → LightGBM Classifier → Threshold Optimization → Output
```

## Candidate Generation / Blocking Strategy

### Multi-Strategy Inverted Index Blocking
At this scale (~12M total records), pairwise comparison is infeasible. We use a multi-strategy blocking approach with inverted indices:

1. **Name Token Blocking**: Inverted index on normalized name tokens (frequency-filtered at 50K max)
2. **First Token Blocking**: Strong signal from the leading word in business names
3. **Name Prefix Blocking**: 5-character prefix matching for catching abbreviation variants
4. **Address Token Blocking**: Top 5 address tokens for geographic co-location signal
5. **Country Bonus Scoring**: Same-country candidates receive a score boost (but no hard filter — France entities must not be dropped)

Each strategy assigns a weighted score to candidates. The top-K candidates (K=30 per source) are selected as the final blocking set.

**Design Decision**: We deliberately do NOT hard-filter by country because:
- France appears only in the test set
- Hard-filtering would silently drop all French entities
- Instead, country match is used as a scoring bonus

### Blocking Quality
- Target recall ceiling: ≥95% on training validation
- Reduction ratio: From ~5M potential pairs per S1 entity to ~60 candidates

## Model Architecture and Feature Engineering

### Features (23 total)
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
- Binary: same normalized country

#### Length-Based Features (4 features)
- Absolute name/address length difference
- Name/address length ratio (0-1, 1=same length)

#### Token Overlap Features (4 features)
- Name/address token overlap count
- Name/address token overlap ratio (to min set size)

#### Combined Feature (1 feature)
- Weighted combination: 0.6 × name_lev + 0.4 × addr_lev

### Model: LightGBM Classifier
- **License**: MIT (compliant with ≤8B params constraint)
- **Architecture**: Gradient-boosted decision trees
- **Hyperparameters**:
  - n_estimators: 1000
  - learning_rate: 0.05
  - num_leaves: 63, max_depth: 8
  - Subsampling: 80% rows, 80% features
  - Regularization: L1=0.1, L2=1.0
  - class_weight: balanced (handles class imbalance)

### Training Strategy
Given the massive dataset, we use **stratified sampling** for training:
- Sample 50K S1 entities (95% with matches, 5% singletons)
- Generate blocking candidates for sampled entities
- Create positive pairs from ground truth + negative pairs from non-matching candidates
- Negative sampling ratio: 5:1 (max 5 negatives per positive)

### Threshold Optimization
- Sweep thresholds from 0.20 to 0.95 on held-out validation set (10K entities)
- Select threshold maximizing macro-averaged F0.5
- F0.5 is precision-heavy, so optimal threshold is typically >0.5

## Text Normalization
- Unicode NFKD normalization with accent stripping
- Lowercase conversion
- Legal suffix expansion (Corp→Corporation, Pvt→Private, Ltd→Limited, etc.)
- Address abbreviation expansion (Ave→Avenue, Blvd→Boulevard, etc.)
- Careful handling of ambiguous abbreviations (avoids corrupting "St. Louis", "Dr." in names, "FL" for Florida)
- Punctuation removal and whitespace normalization
- Stop word removal for blocking tokens

## Unique Contributions
1. **Scalable Blocking**: Multi-strategy inverted index blocking that handles 12M+ records with O(N) index construction
2. **Frequency-Filtered Indices**: High-frequency tokens (>50K occurrences) are excluded from blocking to prevent memory blowup and reduce false candidates
3. **Open-Set Country Handling**: Country is used as a scoring bonus, not a hard filter — properly handles France entities in the test set that don't appear in training
4. **Chunked Inference**: Test data processed in configurable chunks to manage memory on systems with limited RAM
5. **Stratified Training Sampling**: Smart sampling of training data maintains the singleton/matched ratio while keeping training tractable

## Reproducibility

### Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Full Pipeline Run
```bash
python -m src.pipeline --project-dir . --sample-size 50000 --top-k 30
```

### Output Files
- `output/matching_results.tsv` — Final entity matches (scored on leaderboard)
- `output/candidate_pairs.tsv` — Blocking candidate set (for analysis)

## Results
- **Validation F0.5**: [To be filled after full run]
- **Public Leaderboard F0.5**: [To be filled after submission]

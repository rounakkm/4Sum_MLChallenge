"""Business Entity Resolution — Amazon ML Challenge 2026.

This package implements a complete entity resolution pipeline for matching
business records across multiple data sources using:
  - Multi-strategy inverted-index blocking (token, prefix, n-gram, country)
  - Rich pairwise similarity features (23 features)
  - LightGBM classifier optimized for F0.5

Modules:
  - data: Data loading and file discovery
  - normalize: Text normalization for names, addresses, countries
  - blocking: Candidate generation via inverted-index blocking
  - features: Pairwise similarity feature extraction
  - evaluate: Macro-averaged F0.5 scoring
  - pipeline: End-to-end pipeline orchestration
"""
__version__ = "1.0.0"

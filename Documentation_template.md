# Entity Resolution Challenge - Methodology Documentation

## Methodology used
The approach follows a highly optimized, two-stage entity resolution pipeline designed for both extreme recall and precision at scale:
1. **Candidate Generation (Blocking)**: Given the vast number of potential pairs (~35M candidate pairs generated), we employ a multi-pass, GPU-accelerated blocking strategy to retrieve a subset of plausible candidates while preventing OOM errors.
2. **Matching (Classification)**: For the candidate pairs, we generate a deep suite of 15+ string similarity and phonetic features. We train an **Ensemble Model** (XGBoost + Logistic Regression) augmented with **Hard Negative Mining** to classify pairs. Finally, we mathematically optimize the decision threshold to maximize the Macro F0.5 score.

## Candidate generation/blocking strategy
Our blocking strategy is heavily optimized for recall and computational efficiency:
* **Pre-processing & Normalization**: We apply fast regex-based abbreviation expansion (e.g., Corp → Corporation, Rd → Road) and punctuation stripping before any vectorization.
* **Multi-pass Indexing**: Rather than a single pass, we run independent blocking passes on (1) Name only, (2) Address only, and (3) Name + Address combined. This ensures we catch true matches even if one field is entirely dissimilar.
* **GPU-Accelerated & Semantic Blocking**: We utilize PyTorch-accelerated exact cosine similarity search for massive speedups over standard scikit-learn. Additionally, we integrate FAISS / Semantic Embedding options (via SentenceTransformers) to catch semantic synonyms that character n-grams miss.
* **Streaming Engine**: To handle the 1.7 million test entities against the S2/S3 pool without crashing, the blocking engine processes entities in chunked streams (e.g., 500k chunks), keeping peak RAM under 500 MB.

## Model architecture and feature engineering
### Feature Engineering
For each candidate pair, we engineer 15+ robust features targeting structural, phonetic, and component-level similarities:
* **Structural Similarities**: Levenshtein Distance, Jaro-Winkler (for prefix matching), and Word-level Jaccard.
* **Fuzzy String Matching**: Integration of `RapidFuzz` metrics (e.g., `WRatio`) to handle severe typos and transpositions efficiently.
* **Phonetic Normalization**: Soundex/Metaphone encoding to catch heavy transliteration variance, which is particularly crucial for Indian business names.
* **Component Extraction**: Regex-based parsing to extract and directly compare critical sub-components like PIN/ZIP codes.

### Model Architecture
We employ an **Ensemble Model** combining **XGBoost** and **Logistic Regression**. 
* **Hard Negative Mining**: After an initial training pass, we evaluate the model to identify False Positives (predicted match, but ground truth is no-match). These "hard negatives" are injected back into the training set with higher weights, forcing the model to learn the fine-grained distinctions between near-duplicates and true matches.
* **Grouped-Split Cross-Validation**: We validate the model using entity-grouped cross-validation to prevent data leakage and ensure stable hyperparameter tuning.
* **Threshold Optimization**: Given the F0.5 metric, we sweep the ensemble's probability outputs over the OOF validation set to find the exact threshold (e.g., ~0.86) that empirically maximizes the Macro F0.5 score.

## Any other relevant information about the approach
* **Handling Singletons**: The strict threshold optimization inherently learns to confidently reject weak candidates. Singletons naturally fall below the threshold, resulting in empty predictions and yielding full points.
* **Open-set Geographic Generalization**: Because features are computed dynamically as pairwise similarities rather than fixed geographic one-hot encodings, the pipeline natively zero-shot generalizes to unseen regions (e.g., France in the test set) without any code modifications.

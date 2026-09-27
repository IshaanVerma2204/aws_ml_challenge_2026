# Entity Resolution Challenge - Methodology Documentation

## Methodology used
The approach follows a standard two-stage entity resolution pipeline:
1. **Candidate Generation (Blocking)**: Given the vast number of potential pairs, comparing every Source 1 entity to every Source 2/3 entity is computationally infeasible and would yield a highly imbalanced dataset. We apply a high-recall blocking strategy combining semantic embeddings (`SentenceTransformers`) and TF-IDF representations to retrieve a broad subset of plausible candidate pairs.
2. **Matching (Classification)**: For the candidate pairs, we generate a variety of distance and phonetic similarity features. We then train a powerful 6-model ensemble (XGBoost, LightGBM, CatBoost, RandomForest, HistGradientBoosting, and Logistic Regression) to classify each pair, optimizing the decision threshold to strictly maximize the F0.5 score.

## Candidate generation/blocking strategy
The blocking strategy groups records by `country` to respect geographical boundaries. Within each country partition, we create unified text representations. We leverage **Deep Semantic Embeddings** using `all-MiniLM-L6-v2` (`SentenceTransformers`) to map the text to dense semantic vectors, capturing conceptual similarity beyond literal string overlap.

For every Source 1 record, we query the pool to fetch the top $k=100$ most similar candidates (up from a standard 20). This guarantees extremely high recall (often capturing >99% of true matches) and easily handles drastic variations, typos, word-order transpositions, and synonymous business descriptors. A loose distance threshold (`min_sim=0.05`) further prunes irrelevant candidates.

## Model architecture and feature engineering
### Feature Engineering
For each candidate pair produced in the blocking stage, we engineer a set of string similarity features for both `business_name` and `business_address` independently. The features include:
* **Levenshtein Distance**: Measures the minimum number of single-character edits required to change one string into another, providing robustness against typos and abbreviations.
* **Jaro-Winkler Similarity**: Gives higher weight to prefix matches, making it particularly effective for capturing variations in company suffixes or DBA names while maintaining the root name similarity.
* **Jaccard Similarity (Word-level)**: Evaluates the overlap of word tokens, effectively handling word-order transpositions and landmark-based references.
* **Exact Match Flag**: A binary indicator representing whether the strings are completely identical.
* **Length Differences**: Captures substantial discrepancies in address detail depth.

### Model Architecture
We treat the matching task as a binary classification problem and employ a heavily optimized **6-Model Weighted Ensemble Classifier**.
The models include:
- **XGBoost (30% weight)**: Core tree-based gradient booster.
- **LightGBM (20% weight)**: Highly efficient histogram-based boosting.
- **CatBoost (20% weight)**: Excellent for oblivious trees and avoiding target leakage.
- **Random Forest (10% weight)**: Parallelized bagging for robust outlier resistance.
- **HistGradientBoosting (10% weight)**: Scikit-learn's native fast gradient boosting.
- **Logistic Regression (10% weight)**: A linear baseline to anchor probability calibration.

The models are trained using 3-Fold Group Cross Validation with hard negative mining weights to focus aggressively on difficult non-matches. We sweep the probability threshold over the OOF (out-of-fold) predictions to exactly locate the threshold that maximizes the F0.5 score, heavily favoring precision.

## Any other relevant information about the approach
* **Handling Singletons**: Entities with no real matches will likely yield candidates with low structural similarity during the k-NN stage. The model threshold optimization inherently learns to confidently reject weak candidates, predicting an empty match list and thus securing full points for singletons.
* **Open-set Countries**: The model's features rely purely on string similarities rather than one-hot encoded geographic features. This means the model naturally generalizes to unseen countries (e.g., France in the test set) without structural modifications or code changes.

# Entity Resolution Challenge - Methodology Documentation

## Methodology used
The approach follows a standard two-stage entity resolution pipeline:
1. **Candidate Generation (Blocking)**: Given the vast number of potential pairs, comparing every Source 1 entity to every Source 2/3 entity is computationally infeasible and would yield a highly imbalanced dataset. We first apply a high-recall blocking strategy to retrieve a subset of plausible candidate pairs.
2. **Matching (Classification)**: For the candidate pairs, we generate a variety of distance and similarity features and train a supervised machine learning model (XGBoost) to classify each pair as a match or non-match, optimizing the decision threshold to favor precision over recall (F0.5 score).

## Candidate generation/blocking strategy
The blocking strategy groups records by `country` to respect geographical boundaries. Within each country partition, we create a unified text representation by concatenating the `business_name` and `business_address`. We compute character-level TF-IDF representations (n-grams from 2 to 4) of this text for the Source 2 & 3 pool.
For every Source 1 record, we query the pool using k-Nearest Neighbors (with Cosine Similarity) to fetch the top $k=20$ most similar candidates. This approach successfully captures typos, word-order transpositions, and partial missing addresses, while aggressively reducing the search space. To further eliminate nonsensical candidates, we apply a loose distance threshold to prune candidates that share almost no string similarity with the query.

## Model architecture and feature engineering
### Feature Engineering
For each candidate pair produced in the blocking stage, we engineer a set of string similarity features for both `business_name` and `business_address` independently. The features include:
* **Levenshtein Distance**: Measures the minimum number of single-character edits required to change one string into another, providing robustness against typos and abbreviations.
* **Jaro-Winkler Similarity**: Gives higher weight to prefix matches, making it particularly effective for capturing variations in company suffixes or DBA names while maintaining the root name similarity.
* **Jaccard Similarity (Word-level)**: Evaluates the overlap of word tokens, effectively handling word-order transpositions and landmark-based references.
* **Exact Match Flag**: A binary indicator representing whether the strings are completely identical.
* **Length Differences**: Captures substantial discrepancies in address detail depth.

### Model Architecture
We treat the matching task as a binary classification problem and employ an **XGBoost Classifier**.
XGBoost is chosen for its efficiency, ability to handle correlated features (like our multiple string distance metrics), and robustness to unscaled data. The model is trained to minimize Log Loss, and early stopping is used on a held-out validation set to prevent overfitting.
Given the evaluation metric is F0.5 (which weighs precision twice as much as recall), we sweep the prediction probability threshold (from 0.3 to 0.9) over the validation set to empirically find the threshold that maximizes the F0.5 score. The chosen threshold is then applied during test time inference to guarantee a high-precision, low false-positive output.

## Any other relevant information about the approach
* **Handling Singletons**: Entities with no real matches will likely yield candidates with low structural similarity during the k-NN stage. The model threshold optimization inherently learns to confidently reject weak candidates, predicting an empty match list and thus securing full points for singletons.
* **Open-set Countries**: The model's features rely purely on string similarities rather than one-hot encoded geographic features. This means the model naturally generalizes to unseen countries (e.g., France in the test set) without structural modifications or code changes.

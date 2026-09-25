# Business Entity Resolution Pipeline

This is an end-to-end Machine Learning pipeline for the Business Entity Resolution Challenge. It uses TF-IDF and k-Nearest Neighbors for blocking/candidate generation, and an XGBoost model built on string similarity features for the final entity matching.

## Requirements

The dependencies are listed in `requirements.txt`. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

## Running the Pipeline

You can run the pipeline using `main.py`. The script requires the path to the dataset directory (which contains `train/` and `test/` subdirectories) and the output directory.

From the `code/business_entity_resolution/src/` directory, run:

```bash
python main.py --data_dir ../../../dataset --output_dir ../../../output
```

This will:
1. Load the training data from `train/`
2. Perform TF-IDF blocking by country to generate candidate pairs
3. Compute string similarity features (Jaro-Winkler, Levenshtein, Jaccard) for names and addresses
4. Train an XGBoost Classifier and optimize the classification threshold for the F0.5 score
5. Load the test data from `test/`
6. Perform blocking and feature engineering on the test data
7. Predict matching entities
8. Output `matching_results.tsv` and `candidate_pairs.tsv` to the specified output directory.

## Validating Output

Once the pipeline finishes, you can validate your output format using the provided utility (assuming `dataset/test` is at the correct path):

```bash
python3 ../../../utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../dataset/test
```

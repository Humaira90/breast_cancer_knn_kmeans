
# Breast Cancer Classification — KMeans + KNN

## Objective
Classify breast cancer diagnoses into malignant (M) and benign (B) using:
- K-Means clustering with k=2 (for exploratory grouping)
- K-Nearest Neighbors classifier with k=5 (for supervised classification)

## Steps performed (in the notebook)
1. Encode `diagnosis` column: M → 1, B → 0.
2. Drop `id` and any `Unnamed` columns.
3. Apply Min-Max normalization to features (excluding `diagnosis`).
4. Split data into training (80%) and testing (20%) sets (stratified).
5. Apply K-Means (k=2) on the scaled features and map clusters to labels using majority vote.
6. Train KNN (k=5) and evaluate accuracy, precision, recall, and F1-score.

## Key results (from the run)
- K-Means cluster mapping: {0: 0, 1: 1}
- K-Means accuracy: 0.9279
- KNN (k=5) on test set:
  - Accuracy: 0.9649
  - Precision: 1.0000
  - Recall: 0.9048
  - F1-score: 0.9500

## How to run in Google Colab
1. Upload `Dataset.csv` to the Colab session (left pane -> Files -> Upload).
2. Upload `breast_cancer_knn_kmeans.ipynb` (or open it in Colab).
3. Run all cells.

## Notes
- The notebook uses `MinMaxScaler` and `stratify` in the split to keep label distribution consistent.
- K-Means is unsupervised; mapping clusters to labels requires majority-vote mapping, which we included.



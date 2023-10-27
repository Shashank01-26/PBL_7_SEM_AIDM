# Filename: generate_sample_dataset.py
from sklearn.datasets import make_classification
import pandas as pd

# Generate a sample dataset with 100 rows and 5 features
n_samples = 100
n_features = 5

# Create a classification dataset with two classes
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)

# Convert the NumPy arrays to a DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
data['target'] = y  # Assuming you have a target variable

# Save the sample dataset to a CSV file named 'sample_data.csv'
data.to_csv('sample_data.csv', index=False)

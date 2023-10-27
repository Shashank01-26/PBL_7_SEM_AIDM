import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import the joblib library

# Load your sample dataset from a CSV file
data = pd.read_csv('sample_data.csv')

# Perform Numerosity Reduction using Sampling (e.g., random sampling)
sampled_data = data.sample(frac=0.5)  # Adjust the fraction as needed

# Separate the target variable (if applicable) and features
X = sampled_data.drop('target', axis=1)  # Change 'target' to your target column name
y = sampled_data['target']

# Perform Discretization using KBinsDiscretizer
n_bins = 5  # You can adjust the number of bins as needed
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
X_discretized = discretizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_discretized, y, test_size=0.2, random_state=42)

# Initialize and train a classification model (e.g., Decision Tree)
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model to a file
model_filename = 'trained_model.pkl'
joblib.dump(classifier, model_filename)

# Optionally, you can save other relevant data or results as needed

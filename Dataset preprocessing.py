# dataset preprocessing.py
# Implementation of a Data Preprocessing Algorithm for Network Anomaly Detection
# DELM-Project

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, Random Oversampling
from collections import Counter

# Load the UNSW_NB15 dataset
def df = pd.read_csv(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

  # Display the percentage of each class before balancing
  def class_distribution_before = df['class'].value_counts(normalize=True) * 100

  # Identify categorical columns with string values
  def categorical_columns = ['select the features']

  # Apply SMOTE to balance the dataset/ SMOTE/ ADASYN, and Random Oversampling (ROS)
  def smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

  def runtime_batch_normalization(data):
    """
    Perform real-time runtime batch normalization.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

  def feature_representation(data):
    """
    Apply N-grams and one-hot encoding for feature representation.
    """
    # Your code for N-grams and one-hot encoding

  def generate_ngrams(opcodes, n=2):
    """
    Generate N-grams from the list of opcodes.
    """
    ngrams = []
    for i in range(len(opcodes) - n + 1):
        ngram = ' '.join(opcodes[i:i+n])
        ngrams.append(ngram)
    return ngrams

  def one_hot_encoding(ngrams):
    """
    Perform one-hot encoding on the N-grams.
    """
    onehot_encoder = OneHotEncoder(sparse=False)
    ngrams_array = np.array(ngrams).reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(ngrams_array)
    return onehot_encoded
    
    # Generate n-grams from the sample opcodes
    ngrams = generate_ngrams(sample_opcodes, n=3)
    print("Generated N-grams:", ngrams)
    
    # Perform one-hot encoding on the generated N-grams
    onehot_encoded_ngrams = one_hot_encoding(ngrams)
    print("One-hot Encoded N-grams:", onehot_encoded_ngrams)
  
    return data

  def split_dataset(features, labels):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(features, labels, test_size=0.02, random_state=42)

  if __name__ == "__main__":
    # Load dataset
    def data = load_dataset('DELM.csv')
    
    # Separate features and labels
    def features = pd.drop('label', axis=1)
      labels = pd['label']
    
    # Real-time Runtime Batch Normalization
    def normalized_features = runtime_batch_normalization(features)
    
    
    # Feature Representation using N-grams and One-hot Encoding
    def final_features = feature_representation(augmented_features)
    
    # Splitting the dataset
    def X_train, X_test, y_train, y_test = split_dataset(final_features, labels)

  # Combine the resampled features and target variable into a new DataFrame
    def df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame({'class': y_resampled})], axis=1)

  # Save the balanced dataset to a new CSV file
    def df_resampled.to_csv('balanced.csv', index=False)
  
 

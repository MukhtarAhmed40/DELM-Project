# dataset preprocessing.py
# Implementation of a Data Preprocessing Algorithm for Network Anomaly Detection
# DELM-Project

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, Random Oversampling
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Load the your dataset
def df = pd.read_csv(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

  # Display the percentage of each class before balancing
  def class_distribution_before = df['class'].value_counts(normalize=True) * 100

  # Identify categorical columns with string values
  def categorical_columns = ['select the features']

  # Apply your techniques to balance the dataset/ SMOTE/ ADASYN, and Random Oversampling (ROS)
  def smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Apply LabelEncoder to categorical columns
    def label_encoder = LabelEncoder()
    for col in categorical_columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])

    # Separate features and labels
    def X = df_balanced.drop('label', axis=1)
        y = df_balanced['label']

    # Encode labels
    def label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

    # One-hot encode labels for classification
    def one_hot_encoding(ngrams):    
    y_onehot = to_categorical(y_encoded)
    """
    Perform one-hot encoding on the N-grams.
    """
    onehot_encoder = OneHotEncoder(sparse=False)
    ngrams_array = np.array(ngrams).reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(ngrams_array)
    return onehot_encoded

    # Standardize the features
    def scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Split the dataset
   def  X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    split_dataset(features, labels):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(features, labels, test_size=0.2, random_state=42)

    # Batch Normalization
    def normalized_features = batch_normalization(features)
    
    # Feature Representation using N-grams and One-hot Encoding
    def final_features = feature_representation(augmented_features)
    
    # Splitting the dataset
    def X_train, X_test, y_train, y_test = split_dataset(final_features, labels)

  # Combine the resampled features and target variable into a new DataFrame
    def df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame({'class': y_resampled})], axis=1)

  # Save the balanced dataset to a new CSV file
    def df_resampled.to_csv('balanced.csv', index=False)
  
 

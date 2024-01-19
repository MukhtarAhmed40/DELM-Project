# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Assuming you have the balanced dataset stored in a variable 
# Load the balanced dataset
data = pd.read_csv('Dataset.csv')

# Data preprocessing
categorical_columns = ['features']

# Apply LabelEncoder to categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    if data[col].dtype == 'object':
        data[col] = label_encoder.fit_transform(data[col])

# Convert the label column to binary (0 for normal, 1 for attack) Binary classification
df_balanced['attack_cat'] = np.where(df_balanced['attack_cat'] == 'Normal', 0, 1)

# Separate features and labels
X = data.drop('label', axis=1)
y = data['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode labels
y_onehot = to_categorical(y_encoded)

# Label encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Random Oversampling (commented out for now)
oversampler = RandomOverSampler(random_state=42)
 X_train, y_train = oversampler.fit_resample(X_train, y_train)

# Apply ADASYN for oversampling and RandomUnderSampler for undersampling
ada = ADASYN(sampling_strategy='auto', random_state=42)  # You can adjust 'auto' based on your needs
rus = RandomUnderSampler(random_state=42)

try:
    X_train_resampled, y_train_resampled = ada.fit_resample(X_train, y_train)
except ValueError:
    print("ADASYN could not generate samples with the provided ratio settings. Adjusting sampling strategy.")
    # Adjust the sampling strategy based on the class distribution
    class_counts = Counter(y_train)
    min_class = min(class_counts, key=class_counts.get)
    sampling_strategy = {min_class: class_counts[min_class] * 5}  # Adjust 5 as needed
    ada = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = ada.fit_resample(X_train, y_train)

X_train_resampled, y_train_resampled = rus.fit_resample(X_train_resampled, y_train_resampled)

# Apply L1 regularization to MLP weights
alpha = 0.01  # adjust the regularization strength
for i in range(len(mlp_weights)):
    mlp_weights[i] = mlp_weights[i] - alpha * np.sign(mlp_weights[i])

# Set the modified weights back to the MLP model
dbn.named_steps['mlp'].coefs_ = mlp_weights

# DBN Model
def create_dbn_model():
    # Create and return a DBN model
    # Replace the following lines with your DBN model creation code
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)
    return model

# RBM Model
def create_rbm_model():
    # Create and return an RBM model
    # Replace the following lines with your RBM model creation code
    model = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=100, random_state=0, verbose=True)
    return model


# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(128, input_shape=(X_train_resampled.shape[1], 1), activation='relu'))
lstm_model.add(Dense(len(label_encoder.classes_), activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Reshape data for LSTM model input
X_train_lstm = X_train_resampled.reshape((X_train_resampled.shape[0], X_train_resampled.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Fit the LSTM model
lstm_model.fit(X_train_lstm, y_train_resampled_onehot, epochs=100, batch_size=64, validation_data=(X_test_lstm, y_test_onehot))


# DNN Model
def create_dnn_model():
    # Create and return a DNN model
    # Replace the following lines with your DNN model creation code
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# FDNN Model
def create_fdnn_model():
   model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification, so 1 output neuron with sigmoid activation
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 
  
# for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train individual models
dbn_model = create_dbn_model()
dbn_model.fit(X_train, y_train)
dbn_features = dbn_model.predict_proba(X_test)

rbm_model = create_rbm_model()
rbm_model.fit(X_train)  # Fit RBM model with training data
rbm_features = rbm_model.transform(X_test)

lstm_model = create_lstm_model()
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_pred = np.argmax(lstm_model.predict(X_test_lstm), axis=1)

dnn_model = create_dnn_model()
dnn_pred = np.argmax(dnn_model.predict(X_test), axis=1)

fdnn_model = create_fdnn_model()
fdnn_model.fit(X_train, y_train)
fdnn_pred = fdnn_model.predict(X_test)

# Ensure that the number of samples is the same for all features
min_samples = min(dbn_features.shape[0], rbm_features.shape[0], lstm_pred.shape[0], dnn_pred.shape[0], fdnn_pred.shape[0])
dbn_features = dbn_features[:min_samples]
rbm_features = rbm_features[:min_samples]
lstm_pred = lstm_pred[:min_samples]
dnn_pred = dnn_pred[:min_samples]
fdnn_pred = fdnn_pred[:min_samples]

# Stack predictions and features for ensemble
ensemble_input = np.hstack((dbn_features[:, 1:], rbm_features, lstm_pred.reshape(-1, 1), dnn_pred.reshape(-1, 1), fdnn_pred.reshape(-1, 1)))

# Train a logistic regression model as the meta-classifier
meta_classifier = LogisticRegression()
meta_classifier.fit(ensemble_input, y_test)

# Make predictions on the test set using the ensemble
ensemble_pred = meta_classifier.predict(ensemble_input)

# Evaluate the ensemble model
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_classification_report = classification_report(y_test, ensemble_pred)

print("Ensemble Accuracy:", ensemble_accuracy)
print("Ensemble Classification Report:\n", ensemble_classification_report)





































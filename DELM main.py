# main.py
# Implementation of the DELM for network anomaly detection 
# DELM-Project

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.metrics import accuracy_score, classification_report

# Regularization and Layers
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam

# Assuming you have the balanced dataset stored in a variable named 
# Load the dataset
data = pd.read_csv('anomaly_detection_dataset.csv')

# Data Preprocessing
# Assuming 'features' contains feature and 'labels' contains corresponding labels
features = data['features']
labels = data['labels']
categorical_columns = ['feature']

# Apply LabelEncoder to categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    if data[col].dtype == 'object':
        data[col] = label_encoder.fit_transform(data[col])

# Convert the label column to binary (0 for normal, 1 for attack) - for the binary classification
data['label'] = np.where(data['label'] == 'Normal', 0, 1)

# Separate features and labels
X = df_balanced.drop('label', axis=1)
y = df_balanced['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode labels for multi-class classification
y_onehot = to_categorical(y_encoded)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Batch Normalization and feature agregation
from sklearn.utils import shuffle

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# DELM classification model with 5 layers and additional techniques
layer_sizes = [64, 128, 256, 512, 1024]

# Iterate over different n-gram values (1-gram, 2-gram, 3-gram)
for ngram in range(1, 4):
    print(f"\nResults for {ngram}-gram")

#5 layers and additional techniques
for i, layer_size in enumerate(layer_sizes):
    print(f"\nResults for Layer {i + 1} (Size: {layer_size})")

    # Build model up to the current layer
    model = Sequential()
    for j in range(i + 1):
        model.add(Dense(layer_sizes[j], input_dim=X_train.shape[1], activation='relu'))
        model.add(BatchNormalization())  # Batch normalization layer
        model.add(Dropout(0.2))

    model.add(Dense(len(label_encoder.classes_), activation='softmax'))


# DBN pipeline with BernoulliRBM and MLPClassifier (Deep Belief Network)
dbn = Pipeline(
    steps=[
        ('rbm', BernoulliRBM(random_state=42, verbose=True)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42))
    ]
)

# Train the DBN model
dbn.fit(X_train, np.argmax(y_train, axis=1))

# Access the MLP weights
mlp_weights = dbn.named_steps['mlp'].coefs_

# Apply L1 regularization to MLP weights
alpha = 0.01  # adjust the regularization strength
for i in range(len(mlp_weights)):
    mlp_weights[i] = mlp_weights[i] - alpha * np.sign(mlp_weights[i])

# Set the modified weights back to the MLP model
dbn.named_steps['mlp'].coefs_ = mlp_weights

# Evaluate the DBN model
y_pred_dbn = dbn.predict(X_test)

# Classification model with L1 regularization
model = Sequential()
model.add(Dense(64, input_dim=X_test.shape[1], activation='relu', kernel_regularizer=l1(0.01)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_regularizer=l1(0.01)))
model.add(Dropout(0.2))

'''''''''''''

# DELM classification model with 5 layers and Batch Normalization
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Multi classification, so using 'softmax' activation
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Binary classification, so using 'sigmoid' activation
model.add(Dense(1, activation='sigmoid'))  

# Compile the model - for multi-classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Compile the model with Adam optimizer- for binary classification
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Convert one-hot encoded labels back to original labels
y_test_classes = np.argmax(y_test, axis=1)

# Convert class_names elements to strings
#class_names = class_names.astype(str)
class_names = [str(class_name) for class_name in class_names]

# DELM Model Accuracy and Loss training

# Save the best weights during training
best_weights = ModelCheckpoint(filepath='best_weights.h5', save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max', verbose=1)

# Training loop
best_val_accuracy = 0.0
for iteration in range(300):
    print(f"Iteration {iteration + 1}/300")

# Lists to store training history
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []

# Training loop
best_val_accuracy = 0.0
for iteration in range(2):
    print(f"Iteration {iteration + 1}/2")
    
    # Train the DELM model
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[best_weights], verbose=1)

    # Append training history
    train_loss_history.append(history.history['loss'][0])
    train_accuracy_history.append(history.history['accuracy'][0])

    # Evaluate the AFA-FFDNN model on validation set
    val_loss, val_accuracy = model.evaluate(X_test, y_test, verbose=0)
    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Check if validation accuracy improved
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        model.save('best_model.h5')
        print("Best model saved!")

# Load the best weights back into the model
model.load_weights('best_weights.h5')


# Select the top 20 features
top_20_features = [
    'id', 'sttl', 'ct_state_ttl', 'rate', 'sload', 'dttl', 'dload', 'sbytes', 'dmean', 'dur',
    'tcprtt', 'smean', 'dpkts', 'ct_srv_dst', 'synack', 'dinpkt', 'dbytes', 'sinpkt', 'ct_dst_src_ltm', 'ct_srv_src'
]
y = df['attack_cat']  # Replace 'target_column' with the actual target column name

# Function to build AFA-FFDNN model
def build_model(input_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification, adjust as needed

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, feature_set_size):
    model = build_model(feature_set_size)

# Train the model
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Divide features into sets of 10, 6, and 4
features_10 = top_20_features[:10]
features_6 = top_20_features[:6]
features_4 = top_20_features[:4]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert feature names to column indices
features_10_indices = [df.columns.get_loc(feature) for feature in features_10]
features_6_indices = [df.columns.get_loc(feature) for feature in features_6]
features_4_indices = [df.columns.get_loc(feature) for feature in features_4]

# Train and evaluate models for each set of features
train_and_evaluate(X_train_scaled[:, features_10_indices], X_test_scaled[:, features_10_indices], y_train, y_test, feature_set_size=10)
train_and_evaluate(X_train_scaled[:, features_6_indices], X_test_scaled[:, features_6_indices], y_train, y_test, feature_set_size=6)
train_and_evaluate(X_train_scaled[:, features_4_indices], X_test_scaled[:, features_4_indices], y_train, y_test, feature_set_size=4)

# Print and Display classification report

   

# KL-divergence during the training dataset

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def calculate_kl_divergence(original_distribution, generated_distribution):
    original_probs = np.array(list(original_distribution.values())) / sum(original_distribution.values())
    generated_probs = np.array(list(generated_distribution.values())) / sum(generated_distribution.values())

    kl_divergence = entropy(original_probs, generated_probs)
    return kl_divergence

def plot_distributions(original_distribution, generated_distribution):
    labels = list(original_distribution.keys())
    original_probs = np.array(list(original_distribution.values())) / sum(original_distribution.values())
    generated_probs = np.array(list(generated_distribution.values())) / sum(generated_distribution.values())

    width = 0.35
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x - width/2, original_probs, width, label='Original Data')
    ax.bar(x + width/2, generated_probs, width, label='Generated Data')

    ax.set_ylabel('Probability')
    ax.set_title('Class Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# Example usage:
original_data = {
    'Normal': 56000,
    'Generic': 40000,
    'Exploits': 33393,
    'Fuzzers': 18184,
    'DoS': 12264,
    'Reconnaissance': 10491,
    'Analysis': 2000,
    'Backdoor': 1746,
    'Shellcode': 1133,
    'Worms': 130
}

generated_data = {
    'Normal': 56000,
    'Backdoor': 56000,
    'Analysis': 56000,
    'Fuzzers': 56000,
    'Shellcode': 56000,
    'Reconnaissance': 56000,
    'Exploits': 56000,
    'DoS': 56000,
    'Worms': 56000,
    'Generic': 56000
}

kl_divergence = calculate_kl_divergence(original_data, generated_data)
print(f"KL Divergence: {kl_divergence}")

plot_distributions(original_data, generated_data)


# Example data for 2000 epochs and x values starting from 0.45
epochs = range(1, 4001)  # Replace with your epoch values from 1 to 2000
kl_divergence_values = [0.5 * np.exp(-0.0030 * epoch) for epoch in epochs]  # Adjust the decay rate as needed

# Plotting the line graph without dots
plt.plot(epochs, kl_divergence_values, linestyle='-')
#plt.title('KL Divergence vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.grid(True)
plt.show()




# CGAN with DELM/FDNN model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

# Define a simple generator and discriminator for cGAN
def build_generator(latent_dim, output_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_dim=latent_dim, activation='relu'))
    model.add(keras.layers.Dense(output_dim, activation='tanh'))
    return model

def build_discriminator(input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(256, input_dim=input_dim, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

# Build cGAN
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    model = keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Define DELM model
def build_ffdnn(input_dim):
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, input_dim=input_dim, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

# Generate synthetic samples using cGAN
def generate_samples(generator, latent_dim, class_label, n_samples):
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    generated_samples = generator.predict([noise, class_label])
    return generated_samples

# Train cGAN to oversample minority class
def train_cgan(generator, discriminator, cgan, minority_data, minority_labels, latent_dim, epochs=10000, batch_size=32):
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_samples = generator.predict([noise, minority_labels])
        real_samples = minority_data[np.random.randint(0, minority_data.shape[0], batch_size)]

        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_samples, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_samples, labels_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels_gan = np.ones((batch_size, 1))

        g_loss = cgan.train_on_batch([noise, minority_labels], labels_gan)

        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Example usage
latent_dim = 100
output_dim = 45  # Adjust based on your feature space
ffdnn_input_dim = output_dim
epochs_cgan = 10000
batch_size_cgan = 32
epochs_ffdnn = 100
batch_size_ffdnn = 32

# Load your dataset (replace this with your dataset loading code)
# Assume minority_data and minority_labels are your minority class samples
# minority_data = ...
# minority_labels = ...

# Build and compile cGAN components
generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator.trainable = False

# Build and compile cGAN
cgan = build_cgan(generator, discriminator)
cgan.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train cGAN to oversample minority class
train_cgan(generator, discriminator, cgan, minority_data, minority_labels, latent_dim, epochs_cgan, batch_size_cgan)

# Generate synthetic samples using trained cGAN
n_samples = len(minority_data)  # Adjust based on how many synthetic samples you want to generate
generated_samples = generate_samples(generator, latent_dim, minority_labels, n_samples)

# Combine original and synthetic samples
augmented_data = np.concatenate((minority_data, generated_samples), axis=0)
augmented_labels = np.concatenate((minority_labels, minority_labels), axis=0)  # Labels can remain the same for synthetic samples

# Shuffle the augmented dataset
augmented_data, augmented_labels = shuffle(augmented_data, augmented_labels)

# Build and compile FFDNN
ffdnn = build_ffdnn(ffdnn_input_dim)
ffdnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train FFDNN on the augmented dataset
ffdnn.fit(augmented_data, augmented_labels, epochs=epochs_ffdnn, batch_size=batch_size_ffdnn, validation_split=0.2)


#t-SNE visulization

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from keras.utils import to_categorical

# Apply t-SNE to the entire dataset
tsne = TSNE(n_components=2, perplexity=5.0, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame for visualization
tsne_df = pd.DataFrame(data=X_tsne, columns=['t-SNE1', 't-SNE2'])
tsne_df['Class'] = y.values

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Class', data=tsne_df, palette='viridis')
plt.title('t-SNE Visualization of Original Data with All Attack Categories')
plt.show()








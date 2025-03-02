from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
import os
import tensorflow as tf

# Define actions, number of sequences, and sequence length if external bata chalena vane(in my case)
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
no_sequences = 37
sequence_length = 37
DATA_PATH = 'D:/myprojects/signlanguageml/data'

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            print(f"Shape of res: {res.shape}")  # Print the shape of each res array
            if len(window) > 0 and res.shape != window[0].shape:
                print(f"Skipping frame {frame_num} in sequence {sequence} for action {action} due to shape mismatch.") # shape difference le garda model train/compialtion ma error aauxa
                continue
            window.append(res)
        if len(window) == sequence_length:
            print(f"Shape of window: {np.array(window).shape}")  # Print the shape of the window list
            sequences.append(window)
            labels.append(label_map[action])

# Filter out sequences with incorrect shapes
filtered_sequences = [seq for seq in sequences if np.array(seq).shape == (sequence_length, 63)]
filtered_labels = [labels[i] for i in range(len(sequences)) if np.array(sequences[i]).shape == (sequence_length, 63)]

# Convert to numpy arrays
try:
    X = np.array(filtered_sequences)
    print(f"Shape of X: {X.shape}")  # Print the shape of the final X array
except ValueError as e:
    print(f"Error converting sequences to numpy array: {e}")
    for i, seq in enumerate(filtered_sequences):
        print(f"Shape of sequence {i}: {np.array(seq).shape}")

y = to_categorical(filtered_labels).astype(int)
print(f"Shape of y: {y.shape}")  # Print the shape of the final y array

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Setup TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback], validation_data=(X_test, y_test))

# Print model summary if evaluate garne vaye/ projects ma use garne vaye
model.summary()

# Save the model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')

#chandey dai le model banayo

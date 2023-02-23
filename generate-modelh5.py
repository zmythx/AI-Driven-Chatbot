import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple Keras model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on some dummy data
X = [[0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]]
y = [0, 1]
model.fit(X, y, epochs=10)

# Save the model to an .h5 file
model.save('model.h5')
# Import necessary libraries

from keras.models import Sequential  # Import the Sequential model for building neural networks
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten  # Import layers for the neural network
from keras.optimizers import Adam  # Import Adam optimizer for model training (Adaptive Moment Estimation)
from keras.preprocessing.image import ImageDataGenerator  # Import data generator for image preprocessing

# Initialize image data generators for training and validation data
train_data_gen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values of training data to the range [0, 1]
validation_data_gen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values of validation data to the range [0, 1]

# Preprocess train images using the data generator
train_generator = train_data_gen.flow_from_directory(
    'data/train',  # Directory containing training images
    target_size=(48, 48),  # Resize images to 48x48 pixels
    batch_size=64,  # Batch size for training
    color_mode="grayscale",  # Convert images to grayscale
    class_mode='categorical'  # Use categorical labels for classification
)

# Preprocess validation images using the data generator
validation_generator = validation_data_gen.flow_from_directory(
    'data/test',  # Directory containing validation images
    target_size=(48, 48),  # Resize images to 48x48 pixels
    batch_size=64,  # Batch size for validation
    color_mode="grayscale",  # Convert images to grayscale
    class_mode='categorical'  # Use categorical labels for classification
)

# Create the CNN (Convolutional Neural Network) model structure
emotion_model = Sequential()

# Add Convolutional layers with activation functions
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))  # Apply dropout to reduce overfitting

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())  # Flatten the output of the previous layers
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))  # Output layer with 7 classes (emotions)

# Compile the model with categorical cross-entropy loss and Adam optimizer
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the model using the data generators
emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=50, #reduce epochs to reduce time
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size
)

# Save the model structure in a JSON file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights in .h5 file
emotion_model.save_weights('emotion_model.h5')

# MNIST-Handwritten-Digit-Classification

MNIST dataset:
MNIST is a collection of handwritten digits from 0-9. Image of size 28 X 28

Requirements
Python 3.5 +
Scikit-Learn (latest version)
Numpy (+ mkl for Windows)
Matplotlib
Introduction
MNIST contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels, and centered to reduce preprocessing and get started quicker.

Keras is a high-level neural network API focused on user friendliness, fast prototyping, modularity and extensibility. It works with deep learning frameworks like Tensorflow, Theano and CNTK, so we can get right into building and training a neural network without a lot of fuss.

Description
This is a 5 layers Sequential Convolutional Neural Network for digits recognition trained on MNIST dataset. I chose to build it with keras API (Tensorflow backend) which is very intuitive.

Accuracy
It achieved 99.51% of accuracy with this CNN trained on a GPU, which took me about a minute. If you dont have a GPU powered machine it might take a little longer, you can try reducing the epochs (steps) to reduce computation.

It achieved 98.15% of accuracy on test set of this CNN model trained on GPU.

---------------------------------------------------------------------------------------------------------------------------------------
1. Core Building Blocks
Neurons, weights, biases, and activation functions
Neurons: Each layer of the CNN consists of neurons that process inputs and pass outputs to the next layer. The convolutional and dense layers in the code define these neurons.
python
Copy code
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Dense(10, activation='softmax'))
Weights and Biases: These are automatically initialized and adjusted during training using backpropagation. Convolutional and dense layers maintain weights.
Activation Functions:
ReLU (Rectified Linear Unit): Used in the hidden layers (Conv2D) to introduce non-linearity.
python
Copy code
model.add(Conv2D(64, kernel_size=3, activation='relu'))
Softmax: Used in the output layer for multiclass classification. It converts logits to probabilities.
python
Copy code
model.add(Dense(10, activation='softmax'))
Loss Functions
The categorical cross-entropy loss is used because the task involves multiclass classification.
python
Copy code
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Backpropagation and Gradient Descent
Backpropagation: The model uses backpropagation to compute gradients for weights and biases by minimizing the loss function.
Gradient Descent: The Adam optimizer implements an advanced version of gradient descent to update weights efficiently.
python
Copy code
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
2. Basics of Model Training
Data Preprocessing
Normalization: Each pixel value is implicitly scaled by reshaping the images into a range suitable for the model, though explicit normalization isn't shown in this code.
Train-Test Split: The MNIST dataset is pre-split into training and testing sets:
python
Copy code
(X_train, y_train), (X_test, y_test) = mnist.load_data()
Overfitting vs. Underfitting
The validation loss and accuracy are monitored during training to check for signs of overfitting (when validation accuracy stops improving or validation loss increases).
python
Copy code
hist = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=10)
Regularization
While L1 or L2 regularization isn't explicitly applied here, the use of dropout layers or additional constraints can prevent overfitting in an extended version of the model.
3. Workflow Example
Step 1: Load and preprocess a small dataset
The MNIST dataset is loaded and reshaped for compatibility with the CNN:
python
Copy code
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
Step 2: Build a simple neural network
The CNN is defined with two convolutional layers, one pooling layer, and a dense output layer:
python
Copy code
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
Step 3: Train and evaluate the model
The model is trained for 10 epochs using the training data and evaluated on the test set:
python
Copy code
hist = model.fit(X_train, y_train_one_hot, validation_data=(X_test, y_test_one_hot), epochs=10)
Step 4: Visualize metrics
The training and validation accuracy/loss metrics are returned in the hist object and can be visualized using libraries like matplotlib:
python
Copy code
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
Practical Applications in This Code
Neurons, weights, biases, and activation functions are used to build the network layers and process the input data.
Loss functions and gradient descent ensure the model learns effectively during training.
Data preprocessing prepares the raw dataset into a suitable format for training.
The train-test split helps in evaluating the model's performance.
Monitoring overfitting ensures the model generalizes well on unseen data.

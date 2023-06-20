'''
Training the Neural Network.

The neural Network is trained on 60000 examples of images containing handwritten digits.
It is tested on 10000 images.

Dataset downloaded from Kaggle (https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download) 
under the Creative Commons license.
'''

# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from NeuralNetwork import NN
from NeuralNetwork import Functions
from NeuralNetwork import Metrics
from NeuralNetwork import Utils



# load datasets
dataset_train = pd.read_csv('mnist_train.csv')
dataset_test = pd.read_csv('mnist_test.csv')


# get X and y values
X_train  = dataset_train.iloc[:, 1:].values
y_train_labels  = dataset_train.iloc[:, 0].values

X_test  = dataset_test.iloc[:, 1:].values
y_test_labels  = dataset_test.iloc[:, 0].values


# get y values to correspond to 10 outputs of the neural network
def labels_to_outputs(labels:np.ndarray, num_outputs):
    y_train=None
    for value in labels:
        outputs = np.zeros((1,num_outputs), dtype=np.float32)
        outputs[0, value] = 1.
        if isinstance(y_train, np.ndarray):
            y_train = np.append(y_train, outputs, axis=0)
        else:
            y_train = outputs

    return y_train


# reverse outputs to labels
def outputs_to_labels(batch_outputs:np.ndarray):
    labels = None
    for batch in batch_outputs:
        for output in batch:
            label = np.argmax(output)
            if isinstance(labels, np.ndarray):
                labels = np.append(labels, label)
            else:
                labels = np.array([label])

    return labels



y_train = labels_to_outputs(y_train_labels, 10)
y_test = labels_to_outputs(y_test_labels, 10)


# normalize X_train and X_test data
norm = Normalizer()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)




# process dataset into batches
def get_batch(sample_size:int, X:np.ndarray, y:np.ndarray):
    '''
    Processes datasets into batches and returns a tuple of 3D numpy arrays.
    '''
    X_batch = []
    y_batch = []
    remaining_samples = X.shape[0] % sample_size

    samples = X.shape[0] - remaining_samples

    for idx in np.arange(0, samples, sample_size):
        X_batch.append(X[idx : idx + sample_size, :])
        y_batch.append(y[idx : idx + sample_size, :])

    if remaining_samples > 0:
        X_batch.append(X[samples : , :])
        y_batch.append(y[samples : , :])

    return np.array(X_batch), np.array(y_batch)


# create neural network
layer1 = NN(inputs=784, outputs=256, activation='relu')
layer2 = NN(inputs=256, outputs=512, activation='sigmoid')
layer3 = NN(inputs=512, outputs=256, activation='sigmoid')
layer4 = NN(inputs=256, outputs=10, activation='softmax')

back_prop = NN.Backward_Propagation(loss='MCE') # backpropagation class
optim = NN.Optimizer(optimizer='ADAM', alpha=0.001) # optimizer

# utility classes
f = Functions()
m = Metrics()
utils = Utils()



# get data batches
batch_size = 60000
X_train_batch, y_train_batch = get_batch(batch_size, X_train, y_train)


# train the neural network

epochs = 1500

for epoch in range(epochs):

    # forward propagate
    y_hat = layer1.forward(X_train_batch)
    y_hat = layer2.forward(y_hat)
    y_hat = layer3.forward(y_hat)
    y_hat = layer4.forward(y_hat)
    
    # clear gradients
    back_prop.clear_gradients()
    # back propagate
    back_prop.backward(y_train_batch)

    # update weights
    optim.step()

    # get the labels from y_hat
    y_hat = outputs_to_labels(y_hat)
    accuracy = m.accuracy(y_hat, y_train_labels)
    avg_accuracy = m.moving_average(accuracy, 5)

    print(f'Epoch: {epoch} --------- Accuracy: {avg_accuracy * 100} %')


# save the trained weights
utils.save_weights(NN.WEIGHTS, 'mnist_weights')




# test the neural network against the test set

# get entire test set in a batch
X_test_batch, y_test_batch = get_batch(10000, X_test, y_test)

# forward propagate
y_hat = layer1.forward(X_test_batch)
y_hat = layer2.forward(y_hat)
y_hat = layer3.forward(y_hat)
y_hat = layer4.forward(y_hat)

# get the labels from y_hat
y_hat = outputs_to_labels(y_hat)

accuracy = m.accuracy(y_hat, y_test_labels)

print(f'\n\n*******  Test Set Accuracy: {accuracy * 100} %  *******\n\n')



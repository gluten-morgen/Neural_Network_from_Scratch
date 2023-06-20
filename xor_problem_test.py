'''
Testing the neural network to approximate a XOR Gate.


The trained weights have an accuracy of 100% when tested on random samples.

'''

# importing the libraries
import numpy as np
from NeuralNetwork import NN


# create the data using random values
def get_samples(sample_size):

    X = []
    Y = []
    for _ in range(sample_size):
        a = np.random.randint(0, 1000)
        b = np.random.randint(0, 3000)

        a = int(a % 2 == 0 or a % 7 == 0)
        b = int(b % 3 == 0 or b % 5 == 0)

        y1 = a ^ b

        x = [a, b]
        y = [y1]

        X.append(x)
        Y.append(y)

    return np.array([X]), np.array([Y])



# return batches of data
def get_batch(batches, batch_size):
    X_batch, Y_batch = get_samples(batch_size)
    for _ in range(batches - 1):
        X_batch = np.concatenate([X_batch, get_samples(batch_size)[0]], axis=0)
        Y_batch = np.concatenate([Y_batch, get_samples(batch_size)[1]], axis=0)

    return X_batch, Y_batch
        

# initialize the neural netowrk layers
layer1 = NN(2, 5, 'relu', load_weights=True, filename='xor_weights')
layer2 = NN(5, 1, 'sigmoid')


# get a single batch of samples

samples = 10

X, Y = get_samples(samples)

y_hat = layer1.forward(X)
y_hat = layer2.forward(y_hat)

y_pred = y_hat > 0.5

X = X.squeeze()
y_pred = y_pred.squeeze()

for inputs, output in zip(X, y_pred):
    print(f'\ninput: {tuple(inputs)} -------------- output: {int(output)}\n')




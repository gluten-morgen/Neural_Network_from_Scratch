'''
Training the neural network to approximate a XOR Gate.

'''

# importing the libraries
import numpy as np
from NeuralNetwork import NN
from NeuralNetwork import Metrics
from NeuralNetwork import Functions
from NeuralNetwork import Utils
from NeuralNetwork import Visualization



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

b_prop = NN.Backward_Propagation(loss='MSE')
optim = NN.Optimizer(optimizer='SGDM', lr=0.001)

metrics = Metrics()
f = Functions()
utils = Utils()
vis = Visualization()

# generate data based on batch size and number of batches
batch_size = 2000
batches = 10
epochs = 20
X, Y = get_batch(batches, batch_size)

error_list = []

for i in range(epochs):
    y_hat = layer1.forward(X)
    y_hat = layer2.forward(y_hat)

    b_prop.clear_gradients()
    b_prop.backward(Y)
    optim.step()
    
    error_list.append(f.MSE(y_hat, Y))
    print(f.MSE(y_hat, Y))


# get a single batch of 1000 samples
X, Y = get_samples(1000)

y_hat = layer1.forward(X)
y_hat = layer2.forward(y_hat)

y_pred = y_hat > 0.5

print('Accuracy:', metrics.accuracy(y_pred.squeeze(), Y.squeeze())*100, '%')

plt = vis.plot_error(error_list)
plt.show()


utils.save_weights(NN.WEIGHTS, 'xor_weights')


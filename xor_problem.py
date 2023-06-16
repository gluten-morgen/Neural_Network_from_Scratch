# %%
import numpy as np
from NeuralNetwork import NN
from NeuralNetwork import Metrics
from NeuralNetwork import Functions
from NeuralNetwork import Utils
from NeuralNetwork import Visualization

# %%
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


def get_batch(batch_size, sample_size):
    X_batch, Y_batch = get_samples(sample_size)
    for _ in range(batch_size - 1):
        X_batch = np.concatenate([X_batch, get_samples(sample_size)[0]], axis=0)
        Y_batch = np.concatenate([Y_batch, get_samples(sample_size)[1]], axis=0)

    return X_batch, Y_batch
        

# %%
layer1 = NN(2, 5, 'relu')
layer2 = NN(5, 1, 'sigmoid')

b_prop = NN.Backward_Propagation(loss='MSE')
optim = NN.Optimizer(optimizer='SGDM')

metrics = Metrics()
f = Functions()
utils = Utils()
vis = Visualization()

# %%
sample_size = 350
batch_size = 3
iterations = 750
X, Y = get_batch(batch_size, sample_size)

error_list = []

for i in range(iterations):
    y_hat = layer1.forward(X)
    y_hat = layer2.forward(y_hat)

    b_prop.clear_gradients()
    b_prop.backward(Y)
    optim.step()

    error_list.append(f.MSE(y_hat, Y))


# %%
X, Y = get_samples(1000)

y_hat = layer1.forward(X)
y_hat = layer2.forward(y_hat)

print('Accuracy:', metrics.accuracy(y_hat, Y)*100, '%')

plt = vis.plot_error(error_list)
plt.show()


utils.save_weights(NN.WEIGHTS, 'xor_weights')


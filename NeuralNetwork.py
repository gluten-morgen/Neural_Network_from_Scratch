'''
Neural Network
---

Neural Network using Numpy arrays. Not supported on Interactive Python Notebooks.



class NN: Initializes a single neural network layer with specified inputs and outputs.

class NN.Backpropagation: Contains the backpropagation algorithm to calculate gradients.

class NN.Optimizer: Contains the algorithms necessary to optimize gradient descent.

class Functions: Contains activation and loss functions plus their derivatives.

class Metrics: Classes to monitor the performance of the Neural Network.

class Utils: Contains utility functions.


***********

Notes:
-
1. RPROP Algorithm from C. Igel and M. Husken, ‘Improving the Rprop Learning Algorithm’.

2. ADAM algorithm from D. P. Kingma and J. Ba, ‘Adam: A Method for Stochastic Optimization’. arXiv, Jan. 29, 2017. Accessed: May 04, 2023. [Online]. Available: http://arxiv.org/abs/1412.6980


'''
# python ver. 3.9.13


# Importing the libraries
import numpy as np
import concurrent.futures as thread
import math
import pickle
import matplotlib.pyplot as plt 
import warnings



# ------------------- NEURAL NETWORK ------------------

class NN:
        

# _______   _______ CLASS VARIABLES _______   _______


    # Stores all the weights
    WEIGHTS = [np.zeros(())]

    # Stores the weight deltas for all batches after backpropagation
    WEIGHTS_CURR_DELTAS_BATCH = None
    WEIGHTS_PREV_DELTAS_BATCH = None

    # Stores the weighted net sums and outputs in all batches
    Z_BATCH = None
    A_BATCH = None

    # Stores all the activation functions for every layer
    ACTIVATION=['']

    # Stores the batches of gradients
    GRADIENTS_BATCH = [[np.zeros(())]] # store a zero numpy array to prevent error in clear_gradients()
    # Stores the previous batch of gradients
    GRADIENTS_PREV_BATCH = None
    
    # Layer Counter
    LAYER_CNTR = 1

    # Total layers starting from 1
    LAYERS = 1

    # Batch size
    BATCH_SIZE = None

    # Bias Node
    BIAS = False

    # Load saved weights flag
    LOAD_WEIGHTS = False

# _______   _______   _______   _______   _______   _______
    
    def __init__(self, inputs:int, outputs:int, activation='relu', seed=42, bias=False, load_weights=False, filename=''):
        '''
        # Class NN
        Create and initialize a Neural Network layer.

        ## Parameters

        inputs : number of inputs

        outputs : number of outputs

        activation : activation function (relu, sigmoid, tanh, softmax)

        seed : random seed

        bias : add bias node

        load_weights : load weights from a saved '.pkl' file in the current working directory

        filename : Name of the '.pkl' file containing trained weights. File extension is not required. 
        '''
        self.F = Functions()
        self.utils = Utils(functions=self.F)

        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation.lower()

        self.z = np.array([])
        self.a = np.array([])

        # set bias flag and load_weights flag at first initialization only
        if NN.LAYERS == 1:
            # set if bias node is added
            NN.BIAS = bias
            # set if load_weights is true
            NN.LOAD_WEIGHTS = load_weights

        if NN.BIAS:
            # add the bias node to num of inputs
            self.inputs += 1

        # set the seed value
        np.random.seed(seed)
        
        # initialize random weights
        self.weights = np.random.randn(self.outputs, self.inputs)

        if not NN.LOAD_WEIGHTS:
            NN.WEIGHTS.append(self.weights)

        # load saved weights on the first initialization of the class
        if NN.LAYERS==1 and NN.LOAD_WEIGHTS:
            NN.WEIGHTS = self.utils.load_weights(filename)

        NN.ACTIVATION.append(self.activation)

        # update layers count
        NN.LAYERS += 1



    def calc_feedforward(self, inputs:np.ndarray):
        '''Forward propagate a single layer for a batch of samples'''

        # the weighted linear sum of inputs and bias 
        z = np.dot(self.weights, inputs.T)

        # get activation function
        activation_func = self.utils.get_activation_func(self.activation)
        # output of layer after activation function
        a = activation_func(z)

        return z.T, a.T
    

    def forward_samples(self, inputs:np.ndarray, layer_idx:int, batch_idx:int):
        '''
        Forward propagates and stores previous values for backward propagation
        Forward propagation for n samples in a single batch.

        inputs = numpy array with n inputs, and m samples of shape (m, n).

        layer_idx = the layer number to propagate forward.

        batch_idx = batch number to 'label' the returned output to corresponding batch.
        '''
        if NN.BIAS:
            biases = np.ones((inputs.shape[0], 1))
            # add the bias node
            inputs = np.concatenate([inputs, biases], axis=1)
        # get the weights
        self.weights = NN.WEIGHTS[layer_idx]

        # forward propagate
        z, a = self.calc_feedforward(inputs)

        # return the net sums, activated outputs, and batch index
        return z, a, batch_idx
    

    def forward(self, batch:np.ndarray):
        '''
        Forward propagation.
        
        batch = numpy array of k batches having n inputs and m samples of shape (k, m, n).

        Returns numpy array having k batchs of n outputs amd m samples of shape (k, m, n).
        '''
        # initialize batch lists if forward() is called for the first time or at the beginning of an epoch
        if not NN.A_BATCH or not NN.Z_BATCH or (NN.LAYER_CNTR > NN.LAYERS - 1):
            # get batch size
            NN.BATCH_SIZE = batch.shape[0]

            NN.A_BATCH = [[np.zeros(())] for _ in range(NN.BATCH_SIZE)]
            NN.Z_BATCH = [[np.zeros(())] for _ in range(NN.BATCH_SIZE)]


        # reset layer counter if num of layers are reached
        NN.LAYER_CNTR = 1 if NN.LAYER_CNTR > NN.LAYERS - 1 else NN.LAYER_CNTR

        with thread.ThreadPoolExecutor() as executor:
            # Submit batch processing tasks to the thread pool
            output_batch = [executor.submit(self.forward_samples, batch[i], NN.LAYER_CNTR, i) for i in range(NN.BATCH_SIZE)] 

            # Retrieve results from completed tasks
            outputs = [task.result() for task in thread.as_completed(output_batch)]

        if NN.LAYER_CNTR == 1:
            for idx in range(NN.BATCH_SIZE):
                NN.A_BATCH[idx][0] = batch[idx]

        node_outputs = []
        for output in outputs:
            z, a, idx = output

            NN.Z_BATCH[idx].append(z)
            NN.A_BATCH[idx].append(a)

            node_outputs.append(a)
        
        NN.LAYER_CNTR += 1
        return np.array(node_outputs)

 

# --------------------- BACKWARD PROPAGATION -------------------------#




    class Backward_Propagation:

        def __init__(self, loss='MSE'):
            '''
            Backpropagation class.

            - MSE = Mean Squared Error Loss
            - MCE = Multi-class Cross Entropy Loss 
            '''
            self.F = Functions()
            self.utils = Utils(functions=self.F)

            # store the loss function selection
            self.loss = loss



        
        def get_node_deltas(self, layer_idx:int, A:np.ndarray, Z:np.ndarray, y=np.array([[]]), 
                            next_deltas=np.array([[]]), outputNode=False):
            '''
            Returns the node deltas for the given layer as a numpy array of shape (samples, nodes).
            '''
            delta = np.array([])
 
            # if the node is an output node, get the partial derivative of the loss function
            if outputNode:
                dE_da = self.utils.get_diff_loss_func(self.loss)(A[layer_idx], y) 
                da_dz = self.utils.get_diff_activation_func(NN.ACTIVATION[layer_idx])(Z[layer_idx])

                delta = dE_da * da_dz

                return delta

            # the node deltas for the inner (hidden) layers
            if layer_idx > 0:
                out_weight = NN.WEIGHTS[layer_idx+1]
                # remove bias weight to calculate deltas if bias exists
                if NN.BIAS:
                    out_weight = out_weight[:, :-1]

                s = np.dot(next_deltas, out_weight)

                da_dz = self.utils.get_diff_activation_func(NN.ACTIVATION[layer_idx])(Z[layer_idx])

                delta = s * da_dz
            
            return delta
        
        


        def get_gradients(self, y:np.ndarray, A:np.ndarray, Z:np.ndarray):
            '''
            Calculates the gradients for all the weights using backpropagation
            '''
            deltas = [np.zeros(()) for _ in range(NN.LAYERS)]
            node_deltas = [np.zeros(()) for _ in range(NN.LAYERS)]

            # get the node deltas layer by layer
            for layer_num in reversed(range(NN.LAYERS)):
                if layer_num == NN.LAYERS-1:
                    node_deltas[layer_num] = self.get_node_deltas(layer_num, A, Z, y=y, outputNode=True)
                else:
                    node_deltas[layer_num] = self.get_node_deltas(layer_num, A, Z, next_deltas=node_deltas[layer_num + 1], outputNode=False)

            # calculate the gradients
            layer_num = NN.LAYERS - 1
            gradients = [np.zeros(()) for _ in range(NN.LAYERS)]
            while layer_num > 0:
                # get the node deltas for the layer
                deltas = node_deltas[layer_num]

                if NN.BIAS:
                    # add a 1 to the previous outputs for the bias node
                    biases = np.ones((A[layer_num-1].shape[0], 1))
                    # add the bias node
                    prev_outputs = np.concatenate([A[layer_num-1], biases], axis=1)
                else:
                    prev_outputs = A[layer_num-1]

                # add the gradiens to the existing gradients for batch learning
                gradients[layer_num] = np.dot(deltas.T, prev_outputs)

                layer_num -= 1

            return gradients



        def clear_gradients(self):
            '''Clears the gradients to 0.'''
            for batch_idx, _ in enumerate(NN.GRADIENTS_BATCH):
                for layer_idx, _ in enumerate(NN.GRADIENTS_BATCH[batch_idx]):
                    NN.GRADIENTS_BATCH[batch_idx][layer_idx] = np.zeros(NN.GRADIENTS_BATCH[batch_idx][layer_idx].shape)




        def backward(self, Y:np.ndarray):
            '''
            Backward propagate to obtain gradients through neural network

            Y = 3D numpy array with k batches, n outputs of m samples having shape (k, m, n)
            '''
            # initialize gradient batch lists for first iteration of backpropagation
            if not NN.GRADIENTS_BATCH or not NN.GRADIENTS_PREV_BATCH:
                NN.GRADIENTS_BATCH = [self.utils.initialize_in_architecture(NN.WEIGHTS) for _ in range(NN.BATCH_SIZE)]
                NN.GRADIENTS_PREV_BATCH = [self.utils.initialize_in_architecture(NN.WEIGHTS) for _ in range(NN.BATCH_SIZE)]


            with thread.ThreadPoolExecutor() as executor:
                # Submit batch processing tasks to the thread pool
                gradient_batch = [executor.submit(self.get_gradients, Y[batch_idx], NN.A_BATCH[batch_idx], NN.Z_BATCH[batch_idx]) for batch_idx in range(len(NN.A_BATCH))]

                # Retrieve results from completed tasks
                gradients = [task.result() for task in thread.as_completed(gradient_batch)]


            for idx, batch in enumerate(gradients):
                NN.GRADIENTS_BATCH[idx] = batch



        


# -------------------------------- OPTIMIZER ----------------------------------#

    
    class Optimizer:

        def __init__(self, optimizer='SGDM', lr=0.09, momentum=0.0001, 
                     eta_negative=0.2, eta_positive=0.9, step_min=1e-8, step_max=0.0001, zero_tolerance=1e-15, step_init=0.33,
                     alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            '''
            Optimizer Class containing optimizers for training.

            - SGDM = Stochastic Gradient Descent with Momentum
            - RPROP = Resilient Backpropagation with Weight Backtracking Algorithm (RPROP+)
            - ADAM = Adam Optimization Algorithm
            '''
            self.utils = Utils(optimizer=self)
            self.optimizer_select = optimizer.upper()

            # initialize SGDM parameters
            # learning rate
            self.lr = lr
            # momentum
            self.momentum = momentum


            # initialize RPROP parameters
            # zero tolerance
            self.zero_tol = zero_tolerance
            # negative and positive eta 
            self.eta_negative = eta_negative
            self.eta_positive = eta_positive
            # step size bounds 
            self.step_min = step_min
            self.step_max = step_max
            # initialize step
            self.step_init = step_init
            # initialize array of steps
            self.grad_step = self.utils.initialize_in_architecture(NN.WEIGHTS, init='ones')
            for layer_idx, _ in enumerate(self.grad_step):
                self.grad_step[layer_idx] *= self.step_init


            # initialize Adam parameters
            # step size
            self.alpha = alpha
            # exponential decay rates for moment estimates (beta)
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            # initialize first and second moment vectors
            self.m = [np.zeros(()) for _ in range(NN.LAYERS)]
            self.v = [np.zeros(()) for _ in range(NN.LAYERS)]
            # initialize timestep
            self.t = 0


        

        def SGDM(self):
            '''
            Calculates the weight deltas using Stochastic Gradient Descent with Momentum.
            '''
            for batch_idx, batch_gradients in enumerate(NN.GRADIENTS_BATCH):
                for layer_idx, layer in enumerate(batch_gradients):
                    NN.WEIGHTS_CURR_DELTAS_BATCH[batch_idx][layer_idx] = (-self.lr * layer) + (self.momentum * NN.WEIGHTS_PREV_DELTAS_BATCH[batch_idx][layer_idx])


        
        def RPROP(self):
            '''
            Calculates the weight deltas using Resilient Backpropagation Algorithm (RPROP).
            '''
            for batch_idx, (batch_gradients, batch_gradients_prev) in enumerate(zip(NN.GRADIENTS_BATCH, NN.GRADIENTS_PREV_BATCH)):

                for layer_idx, layer in enumerate(batch_gradients):
                    if layer_idx == 0:
                        continue # skip over first (input) layer
                    batch_gradients[layer_idx][batch_gradients[layer_idx] < self.zero_tol] = 0.

                    # get the gradient signs - +1 for positive change, -1 for negative change, 0 for no change
                    grad_signs = self.utils.grad_sign_change(batch_gradients_prev[layer_idx], layer)

                    # for gradients with no change
                    for row, col in zip(*np.where(grad_signs > 0)):
                        self.grad_step[layer_idx][row, col] = min(self.grad_step[layer_idx][row, col] * self.eta_positive, self.step_max)
                        NN.WEIGHTS_CURR_DELTAS_BATCH[batch_idx][layer_idx][row, col] = -1 * self.utils.sign(batch_gradients[layer_idx][row, col]) * self.grad_step[layer_idx][row, col]

                    # for gradients with a sign change
                    for row, col in zip(*np.where(grad_signs < 0)):
                        self.grad_step[layer_idx][row, col] = max(self.grad_step[layer_idx][row, col] * self.eta_negative, self.step_min)
                        NN.WEIGHTS_CURR_DELTAS_BATCH[batch_idx][layer_idx][row, col] = -1 * NN.WEIGHTS_PREV_DELTAS_BATCH[batch_idx][layer_idx][row, col]
                        batch_gradients[layer_idx][row, col] = 0

                    # for gradients equal 0.0
                    for row, col in zip(*np.where(grad_signs == 0)):
                        NN.WEIGHTS_CURR_DELTAS_BATCH[batch_idx][layer_idx][row, col] = -1 * self.utils.sign(batch_gradients[layer_idx][row, col]) * self.grad_step[layer_idx][row, col]



        def adam(self):
            '''
            Calculates the weight deltas using the ADAM (Adaptive Moment Estimation) Algorithm.
            '''
            for batch_idx, batch_gradients in enumerate(NN.GRADIENTS_BATCH):
                # increment timestep for every optimization operation
                self.t += 1

                for layer_idx, layer in enumerate(batch_gradients):
                    if layer_idx == 0:
                        continue # skip over first (input) layer
                    self.m[layer_idx] = (self.beta1 * self.m[layer_idx]) + ((1-self.beta1) * layer) # update biased first moment estimate
                    self.v[layer_idx] = (self.beta2 * self.v[layer_idx]) + ((1-self.beta2) * np.square(layer)) # update biased second raw moment estimate

                    m_hat = self.m[layer_idx] / (1 - math.pow(self.beta1, self.t)) # compute bias-corrected first moment estimate
                    v_hat = self.v[layer_idx] / (1 - math.pow(self.beta2, self.t)) # compute bias-corrected second raw moment estimate

                    NN.WEIGHTS_CURR_DELTAS_BATCH[batch_idx][layer_idx] = -self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)





        def step(self):
            '''
            Updates the weights in the neural network
            '''
            if not NN.WEIGHTS_CURR_DELTAS_BATCH or not NN.WEIGHTS_PREV_DELTAS_BATCH:
                NN.WEIGHTS_CURR_DELTAS_BATCH = [self.utils.initialize_in_architecture(NN.WEIGHTS) for _ in range(NN.BATCH_SIZE)]
                NN.WEIGHTS_PREV_DELTAS_BATCH = [self.utils.initialize_in_architecture(NN.WEIGHTS) for _ in range(NN.BATCH_SIZE)]

            # get the weight changes (deltas)
            self.utils.get_optimizer_func(self.optimizer_select)()

            for batch_deltas in NN.WEIGHTS_CURR_DELTAS_BATCH:
                for layer_idx, layer in enumerate(batch_deltas):
                    NN.WEIGHTS[layer_idx] += layer

            NN.WEIGHTS_PREV_DELTAS_BATCH = [self.utils.copy_parameters(NN.WEIGHTS_CURR_DELTAS_BATCH[i]) for i in range(NN.BATCH_SIZE)]
            NN.GRADIENTS_PREV_BATCH = [self.utils.copy_parameters(NN.GRADIENTS_BATCH[i]) for i in range(NN.BATCH_SIZE)]




# ---------------------------- FUNCTIONS --------------------------------#

    

class Functions:

    def __init__(self):
        '''
        Class containing activation and loss functions plus their derivatives.

        Note: If adding functions here, update the corresponding 'get_' function in NN.Utils.
        '''
        pass

    #-------- ACTIVATION FUNCTIONS -------

    def relu(self, z:np.ndarray):
        '''Rectifier Linear Unit (ReLU) activation function'''
        return np.maximum(np.zeros(z.shape), z)


    def sigmoid(self, z:np.ndarray):
        '''Sigmoid activation function'''
        #suppress warnings
        warnings.filterwarnings('ignore')
        return 1. / (1. + np.exp(-z))
    

    def tanh(self, z:np.ndarray):
        '''tanh activation function'''
        return np.tanh(z)
    

    def softmax(self, z:np.ndarray):
        '''Softmax activation function'''
        #suppress warnings
        warnings.filterwarnings('ignore')
        return np.exp(z) / np.sum(np.exp(z))
    
    #---- LOSS FUNCTIONS ----
    
    def MSE(self, a, y):
        '''Mean Squared Error loss function'''
        sq_diff = np.square(a - y)
        mse = np.mean(sq_diff)
        return mse
    

    def MCE(self, a, y):
        '''Multi-class Cross Entropy Loss function'''
        error = []
        for preds, truths in zip(a, y):
            s = truths * np.log(preds)
            s = -np.sum(s, axis=1)
            error.append(np.mean(s))
        return np.mean(np.array(error))
    
    #-------- ACTIVATION FUNCTION DERIVATIVES -------

    def d_relu(self, z:np.ndarray):
        '''Derivative of the ReLU activation function'''
        return 1. * (z > 0.)
    
    
    def d_sigmoid(self, z:np.ndarray):
        '''Derivative of Sigmoid activation function'''
        return self.sigmoid(z) * (1. - self.sigmoid(z))
    

    def d_tanh(self, z:np.ndarray):
        '''Derivative of the tanh activation function'''
        return 1 - np.tanh(z)**2
    

    def d_softmax(self, z:np.ndarray):
        '''
        Derivative of the Softmax activation function is not implemented, as it is used in d_MCE(). Returns 1.0 by default.
        '''
        if True: return 1.

        # unreachable code that implements the softmax derivative
        rows = np.array([*range(z.shape[0])], dtype=np.int32).reshape((z.shape[0], 1))
        cols = np.array(y.argmax(axis=1), dtype=np.int32).reshape((z.shape[0], 1))
        indices = np.concatenate([rows, cols], axis=1)

        softmax_derivatives = - self.softmax(z)

        probs_int = - softmax_derivatives[indices[:, 0], indices[:, 1]]
        probs_int = probs_int.reshape((softmax_derivatives.shape[0], 1))
        probs_int = np.repeat(probs_int, softmax_derivatives.shape[1], axis=1)

        softmax_derivatives[indices[:, 0], indices[:, 1]] = 1 + softmax_derivatives[indices[:, 0], indices[:, 1]]

        softmax_derivatives = probs_int * softmax_derivatives

        return softmax_derivatives


    #---- LOSS FUNCTION DERIVATIVES ----

    def d_MSE(self, a:np.ndarray, y:np.ndarray):
        '''Derivative of the Mean Squared Error'''
        mse_derivative = 2. * (a - y)
        return np.mean(mse_derivative)
    

    def d_MCE(self, a:np.ndarray, y:np.ndarray):
        '''Derivative of the Multi-Class Cross Entropy Loss function'''
        return a - y
    


# ---------------------------- UTILS --------------------------------#

        
class Utils():

    def __init__(self, functions:Functions=None, optimizer:NN.Optimizer=None):
        '''
        Class containing utility functions.
        '''
        self.F = functions
        self.optim = optimizer


    def get_activation_func(self, activation):
        '''
        Returns the activation function object 
        '''
        if activation == 'relu':
            return self.F.relu
        elif activation == 'sigmoid':
            return self.F.sigmoid
        elif activation == 'tanh':
            return self.F.tanh
        elif activation == 'softmax':
            return self.F.softmax
        else:
            raise Exception('Activation function not implemented.')
        

    def get_diff_activation_func(self, activation):
        '''
        Returns the differentiated activation function.
        '''

        if activation == 'relu':
            return self.F.d_relu
        elif activation == 'sigmoid':
            return self.F.d_sigmoid
        elif activation == 'tanh':
            return self.F.d_tanh
        elif activation == 'softmax':
            return self.F.d_softmax
        else:
            raise Exception('Differentiated activation function not implemented.')
        

    def get_diff_loss_func(self, loss):
        '''
        Returns the differentiated loss function
        '''

        if loss == 'MSE':
            return self.F.d_MSE
        elif loss == 'MCE':
            return self.F.d_MCE
        else:
            raise Exception('Differientiated loss function not implemented.')
        

    def get_optimizer_func(self, optimizer_select):
        '''
        Returns the optimizer function.
        '''
        if optimizer_select == 'SGDM':
            return self.optim.SGDM
        elif optimizer_select == 'RPROP':
            return self.optim.RPROP
        elif optimizer_select == 'ADAM':
            return self.optim.adam
        else:
            raise Exception('Optimizer function not implemented.')
        

    def initialize_in_architecture(self, architecture:list, init='zeros'):
        '''
        Returns a list of np.ndarrays containing zeros or ones in the shape each of the layers in 'architecture'.
        init = 'zeros' or 'ones'
         '''
        init_architecture = []
        if init == 'zeros':
            init_type = np.zeros
        elif init == 'ones':
            init_type = np.ones

        for _, layer in enumerate(architecture):
            init_architecture.append(init_type(layer.shape))
        return init_architecture
    

    def copy_parameters(self, layers:list):
        '''
        Returns a list of np.ndarrays with copies of the parameters of each layer
        '''
        copy_params = []
        for layer in  layers:
            copy_params.append(layer.copy())
        return copy_params
        

    def grad_sign_change(self, grads_prev, grads_curr):
        '''
        Returns the array of gradients with -1 if the sign changed from positive to negative,
        1 if gradients changed from negative to positive, 0 otherwise.
        '''
        # get all the negative changes
        neg_grads = (grads_prev * grads_curr) < 0.
        # get all the positive grads
        pos_grads = (grads_prev * grads_curr) > 0.

        # array of positive, negative, or zero
        sign_arr = (-1 * neg_grads) + pos_grads

        return sign_arr
    

    def sign(self, grad):
        '''
        Returns +1 if argument is positive, -1 if negative, 0 otherwise.
        '''
        if grad > 0.0:
            return 1
        elif grad < 0.0:
            return -1
        else:
            return 0
        

    def save_weights(self, weights:list, filename:str):
        '''
        Pickles weights data to be sved to a file in the current working directory.
        '''
        filename += '.pkl' # add extension to file name

        with open(filename, 'wb') as file:
            pickle.dump(weights, file)


    def load_weights(self, filename:str):
        '''
        Loads the saved weights from a file in the current working directory.
        '''
        filename += '.pkl' # add extension to file name

        with open(filename, 'rb') as file:
            weights = pickle.load(file) 

        return weights

        

# ---------------------------- METRICS --------------------------------#
  


class Metrics:

    def __init__(self):
        '''
        Class to evaluate the performance of a Neural Network.
        '''
        self.m_avg = np.array([])


    def accuracy(self, y_pred:np.ndarray, y:np.ndarray):
        '''Calculates the accuracy of a model based on how often predicted values equal labels.'''
        accuracy = np.ndarray([])
        accuracy = y_pred == y
        return np.mean(accuracy)
    

    def moving_average(self, accuracy:float, sampling=10):
        '''Computes the Moving Average over a stream of accuracy data'''
        self.m_avg = np.append(self.m_avg, [accuracy])
        if len(self.m_avg) > sampling:
            self.m_avg = self.m_avg[sampling:]
        return np.mean(self.m_avg)
    



# ------------------------- VISUALIZATION ----------------------------#


class Visualization:

    def __init__(self):
        pass

    

    def plot_error(self, error_list:list):
        '''
        Plots the error curve from error data gathered during training.
        '''
        x = np.arange(0, len(error_list))
        plt.figure(figsize=(7, 4.3), layout='constrained')
        plt.plot(x, error_list, label='Error') 
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error Curve')
        plt.legend()

        return plt
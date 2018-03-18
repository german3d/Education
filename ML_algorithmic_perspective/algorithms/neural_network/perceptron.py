import os
import itertools
import numpy as np


class Perceptron(object):

    def __init__(self, num_neurons):        
        self.num_neurons = num_neurons
    

    def fit(self, X, y, num_iter=10, eta=0.25):
        
        # adding unit vector for bias term
        inputs = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        num_features = inputs.shape[1]
        
        y = Perceptron._resize_output(y)

        # initialize random weights
        self.weights = np.random.rand(num_features, self.num_neurons)*0.1
        
        # training loop
        for i in range(num_iter):
            print "iteration #: %d"%i 
            self.activations = np.dot(inputs, self.weights)
            self.is_activated = np.where(self.activations>0,1,0)
            
            # break training if all predictions are correct
            if np.sum(np.abs(self.is_activated-y)) == 0:
                print "early stopping - all predictions are correct"
                return            
            self.weights -= eta*np.dot(np.transpose(inputs), self.is_activated-y)
    

    @staticmethod
    def _resize_output(output):
        if len(output.shape)==1:
            return output.reshape((output.shape[0],-1))
        else:
            return output            
    

    def predict(self, X):
        inputs = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        self.activations = np.dot(inputs, self.weights)
        pred = np.where(self.activations>0,1,0)
        return pred
    

    def test_accuracy(self, X_test, y_test):
        y_test = Perceptron._resize_output(y_test)        
        pred_test = self.predict(X_test)
        acc = np.sum(pred_test==y_test, dtype=np.float64) / y_test.shape[0]
        return acc




class MLP(object):
    
    def __init__(self, num_neurons, activation_out="logistic"):
        self.num_neurons = num_neurons
        self.activation_out = activation_out
        self.hidden = []
        self.predictions = None        
        self.train_loss = []
        self.val_loss = []
        self.grads = []
        self._grads_check_ok = None
        
        
    def init_weights(self):        
        dims = [self.input_dim] + self.num_neurons + [self.output_dim]
        weights_shape = [(x+1,y) for x,y in zip(dims[:-1], dims[1:])] # with bias term
        # Xavier initialization
        weights = [np.random.uniform(low=-1.0, high=1.0, size=shape)
                   /np.sqrt(shape[0]) for shape in weights_shape]
        updates = [np.zeros(shape) for shape in weights_shape]
        self.weights = weights
        self.updates = updates
        
        
    @staticmethod
    def add_bias(inputs):
        return np.concatenate((inputs, np.ones((inputs.shape[0],1))), axis=1)
   

    @staticmethod
    def activate(values, func="logistic"):
        if func=="logistic":
            activation = 1.0 / (1.0 + np.exp(-values))
        elif func=="linear":
            activation = values
        elif func=="softmax":
            activation_n = np.exp(values)
            activation_d = np.tile(np.sum(activation_n,axis=1).reshape(-1,1), reps=(1,values.shape[1]))            
            activation = activation_n / activation_d            
        return activation
        
        
    @staticmethod
    def resize_output(output):
            if len(output.shape)==1:
                return output.reshape((output.shape[0],-1))
            else:
                return output
    
    
    @staticmethod
    def compute_accuracy(predictions, targets):
        predictions = np.round(predictions)
        N = predictions.shape[0]
        acc = (predictions==targets).all(axis=1).sum(dtype=np.float) / N
        return acc
    
    
    def compute_loss(self, predictions, targets, validation=False):
        N = predictions.shape[0]
        if self.activation_out=="linear":
            loss = 0.5 * np.sum((predictions-targets)**2) / N
        elif self.activation_out=="logistic":
            loss = 0.5 * np.sum((predictions-targets)**2)
        elif self.activation_out=="softmax":
            loss = np.sum(-targets*np.log(predictions))
        else:
            raise ValueError("Unrecognised activation function")
        if validation:          
            self.val_loss.append(loss)
        else:
            self.train_loss.append(loss)
        
        
    def predict(self, X):
        inputs = MLP.add_bias(X)
        hidden = []
        for W in self.weights[:-1]: # hidden layers
            input_h = np.dot(inputs, W)
            output_h = MLP.add_bias(MLP.activate(input_h))
            hidden.append(output_h)
            inputs = output_h
        input_l = np.dot(inputs, self.weights[-1]) # output layer        
        predictions = MLP.activate(input_l, func=self.activation_out)
        return predictions, hidden
    
    
    def forward_pass(self, X, y, validation=False):
        predictions, hidden = self.predict(X)
        self.compute_loss(predictions, y, validation=validation)
        if not validation:
            self.predictions = predictions
            self.hidden = hidden
        
        
    def back_prop(self, X, y, eta, momentum, update=True):
        N = X.shape[0]
        if self.activation_out=="logistic":
            delta_output = (self.predictions-y)*self.predictions*(1.0-self.predictions)
        elif self.activation_out=="linear":
            delta_output = (self.predictions-y) / N
        elif self.activation_out=="softmax":
            delta_output = (self.predictions-y)
        else:
            raise ValueError("Unrecognised activation function")
        grad_output = np.dot(self.hidden[-1].T, delta_output)

        # gradients of errors w.r.t. matrices of weights
        grads = [grad_output]
        iters = zip([MLP.add_bias(X)] + self.hidden[:-1], self.hidden, self.weights[1:])
        delta_2 = delta_output
        for h_0, h_1, w_1 in iters[::-1]:
            delta_1 = h_1*(1.0-h_1) * np.dot(delta_2, w_1.T)
            grad_0 = np.dot(h_0.T, delta_1[:,:-1])  # excluding bias term
            grads = [grad_0] + grads
            delta_2 = delta_1[:,:-1]
        self.grads = grads        
        if update:
            self.updates = [eta*grad+momentum*upd for grad,upd in zip(self.grads, self.updates)]
            self.weights = [W - upd for W,upd in zip(self.weights, self.updates)]
    
    
    def check_gradients(self, X, y, epsilon=1e-5, delta_thr=1e-6):
        self._init_weights = self.weights
        grads_test = []
        grads_calc = []
        for i in range(len(self.weights)):
            W_h = self._init_weights[i]
            grad_h = self.grads[i]
            num_rows, num_cols = W_h.shape
            coords = itertools.product(range(num_rows), range(num_cols))
            for i_row, i_col in coords:
                init_weight = W_h[i_row,i_col] # weight at initialization
                i_grad = grad_h[i_row,i_col] # calculated gradient on backprop phase

                self.weights[i][i_row,i_col] = init_weight + epsilon
                self.forward_pass(X, y)
                loss_1 = self.train_loss[-1]

                self.weights[i][i_row,i_col] = init_weight - epsilon
                self.forward_pass(X, y)
                loss_2 = self.train_loss[-1]
                
                loss_delta = (loss_1-loss_2) / (2*epsilon)
                
                grads_test.append(loss_delta)
                grads_calc.append(i_grad)
                self.weights[i][i_row,i_col] = init_weight                                                
        
        grads_test = np.array(grads_test)
        grads_calc = np.array(grads_calc)
        delta_abs = np.abs(grads_calc-grads_test)
        delta_norm = delta_abs / np.stack([np.abs(grads_calc), np.abs(grads_test)]).max(axis=0)        
        self._grad_check_delta_mean = delta_norm.mean()
        self._grad_check_delta_max = delta_norm.max()
        self._grad_check_ok = True if self._grad_check_delta_mean < delta_thr else False                
        self.hidden = []
        self.predictions = None
        self.train_loss = []
        self.grads = []
        
        
    def fit(self, X, y, X_val=None, y_val=None, early_stopping=False, max_iter=10, eta=0.25, momentum=0.9):
        self.input_dim = X.shape[1]
        self.output_dim = MLP.resize_output(y).shape[1]
        
        if self.activation_out=="softmax" and self.output_dim<=1:
            raise Exception("Invalid output shape for softmax activation")
        
        # initializing random weights
        self.init_weights()
        
        # resizing output vector to be 2d-array
        y = MLP.resize_output(y)
        
        # check gradients
        self.forward_pass(X, y)
        self.back_prop(X, y, eta, momentum, update=False)
        self.check_gradients(X, y)
        
        for i in range(1, max_iter+1):
            # forward pass
            self.forward_pass(X, y)
            
            # when validation sets are given, check train/validation loss and prevent overfitting
            if X_val is not None and y_val is not None:
                y_val = MLP.resize_output(y_val)
                self.forward_pass(X_val, y_val, validation=True)
                if i==max_iter or i%100==0 or i==1:
                    print "Iter #:{}, train_loss: {:.5f}, validation_loss: {:5f}"\
                    .format(i, self.train_loss[-1], self.val_loss[-1])
                
                # stop training when there is no evidence of validation loss decrease
                if early_stopping and len(self.val_loss)>=5:
                    i_loss_val = self.val_loss[-1]
                    if i_loss_val>self.val_loss[-2]>self.val_loss[-3]>self.val_loss[-4]>self.val_loss[-5]:
                        print "Iter #:{} early stopping, train_loss: {:.5f}, validation_loss: {:5f}"\
                        .format(i, self.train_loss[-1], self.val_loss[-1])
                        return
                
            # otherwise just print training loss progress
            else:
                if i==max_iter or i%100==0 or i==1:
                    print "Iter #:{}, train_loss: {:.5f}".format(i, self.train_loss[-1])
                
            #back propagation
            self.back_prop(X, y, eta, momentum)

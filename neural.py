import numpy as np

class neural_network():

    def __init__(self, n_hidden_layer=1, n_neurons_hl1=16, n_neurons_hl2=8, n_input=4, n_output=3, dropout_prob=None, regularization_factor = 0):
        
        '''
 
        inputs:
                
                n_input ->  number of input features (default = 4)
                n_hidden layer -> neural network number of hidden layers (default=2 )
                n_neurons_hl1 -> number of neurons in hidden layer 1 (default=16)
                n_neurons_hl2 -> number of neurons in hidden layer 2 (default=8)
                n_output -> number of neurons in output layer 3 (default=3)
 
        '''

        self.n_hidden_layer = n_hidden_layer
        self.n_neurons_hl1 = n_neurons_hl1
        self.n_neurons_hl2 = n_neurons_hl2
        self.n_input = n_input
        self.n_output = n_output
        self.dropout_prob = dropout_prob
        self.regularization_factor = regularization_factor


    def hidden_layer_1(self, weights, X_train):

        W1 = weights[0:self.n_input*self.n_neurons_hl1].reshape((self.n_input,self.n_neurons_hl1))
        B1 = weights[self.n_input*self.n_neurons_hl1:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1].reshape((self.n_neurons_hl1,))
        Z1 = X_train.dot(W1)+B1

        # tanh activation function
        
        O1 = np.tanh(Z1)
        
        return O1
        

    def dropout(self,O1):

        for i in range(O1.shape[1]):

            for j in range(O1.shape[0]):

                if np.random.rand() < self.dropout_prob:

                    O1[j,i]=0

                else: pass

        return O1


    def hidden_layer_2(self, O1, weights):
        
        W2 = weights[self.n_input*self.n_neurons_hl1+self.n_neurons_hl1:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2].reshape((self.n_neurons_hl1, self.n_neurons_hl2))
        B2 = weights[self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2+self.n_neurons_hl2].reshape((self.n_neurons_hl2,))
        Z2 = O1.dot(W2)+B2
        
        # tanh activation function
        
        O2 = np.tanh(Z2)

        return O2

    def output(self, O2, weights):
        
        W3 = weights[self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2+self.n_neurons_hl2:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2+self.n_neurons_hl2+self.n_neurons_hl2*self.n_output].reshape((self.n_neurons_hl2, self.n_output))
        B3 = weights[self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2+self.n_neurons_hl2+self.n_neurons_hl2*self.n_output:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2+self.n_neurons_hl2+self.n_neurons_hl2*self.n_output+self.n_output].reshape((self.n_output,))

        Z3 = O2.dot(W3)+B3
        
        # tanh activation function
        
        O3 = np.tanh(Z3)

        return O3


    def softmax(self, logits):

        # softmax activation function for output (multiclass classification)
        
        scores = np.exp(logits)
        probs = scores / np.sum(scores, axis=1, keepdims=True)

        return probs


    def loss(self, probs, y_train):

        one_hot_encoding = np.zeros((len(y_train),3))

        for i in range(len(y_train)):

            if y_train[i]==0:

                one_hot_encoding[i,0]=1
                
            elif y_train[i]==1:
                
                one_hot_encoding[i,1]=1


            elif y_train[i]==2:
                
                one_hot_encoding[i,2]=1

            else: pass

        # negative negative log-likelihood loss
        
        loss_ = np.sum(one_hot_encoding*np.log(probs)*(-1/len(one_hot_encoding)))

        return loss_


    def feedforward(self, weights, X_train, y_train):

        output1 = self.hidden_layer_1(weights, X_train)

        output1 = self.dropout(output1)

        output2 = self.hidden_layer_2(output1, weights)

        output3 = self.output(output2, weights)

        probs = self.softmax(output3)
        
        # compute loss using negative log-likelihood loss
        
        loss_ = self.loss(probs, y_train)

        w1 = weights[self.n_input*self.n_neurons_hl1+self.n_neurons_hl1:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2]
        w2 = weights[self.n_input*self.n_neurons_hl1+self.n_neurons_hl1:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2]
        w3 = weights[self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2+self.n_neurons_hl2:self.n_input*self.n_neurons_hl1+self.n_neurons_hl1+self.n_neurons_hl1*self.n_neurons_hl2+self.n_neurons_hl2+self.n_neurons_hl2*self.n_output]

        # L1 regularization 

        regularization = (1/self.regularization_factor)*(np.mean(np.abs(w1))+np.mean(np.abs(w2))+np.mean(np.abs(w3)))

        # add regularization to the loss

        output = loss_ + regularization
    
        return output 

    def predictions(self, weights, X_train):

        
        output1 = self.hidden_layer_1(weights, X_train)

        output2 = self.hidden_layer_2(output1, weights)

        output3 = self.output(output2, weights)

        probs = self.softmax(output3)
        
        return probs


    def architecture(self):

        print("FeedForward Neural Network developed by Rafael Pavan \n")
        print("Architecture: \n")
        print(f"- {self.n_hidden_layer} Hidden Layers; \n")
        print(f"- Hidden Layer 1: {self.n_neurons_hl1} neurons, activation function: tanh; \n")
        print(f"- Output: {self.n_output} neurons, activation function: softmax; \n")
        print(f"- Loss Function:  Negative Log-Likelihood Loss; \n")
        print(f"- Dropout Probability: {self.dropout_prob} \n")
        print(f"- Regularization: L1, {self.regularization_factor} \n") 


        
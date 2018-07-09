#!/usr/bin/env python

import math
import random
import csv

random.seed(1)
# method for initialize weights randomly between 0-1
def randomise(m, n):
    return [[random.random() for i in range(n)] for j in range(m)]

# sigmoid activation function
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# tanh activation function
def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

# convert letter to index between 0-25 based on ascii
def lettertoindex(l):
    return ord(l)-ord('A')

# convert vector to letter to visualize the result
def vectoletter(v):
    index = v.index(max(v))
    return chr(index+ord('A'))

class MLP(object):
    def __init__(self):
        self.NI = 0
        self.NH = 0
        self.NO = 0
        self.I = []
        self.H = []
        self.O = []
        self.W1 = []
        self.W2 = []
        self.delta_output = []
        self.delta_hidden = []

    # method to create the network
    def initialise(self, NI, NH, NO):
        self.NI = NI + 1 # add bias
        self.NH = NH
        self.NO = NO
        # initialise vector with value 1
        self.I = [1.0] * self.NI
        self.H = [1.0] * self.NH
        self.O = [1.0] * self.NO
        # initialise weights between 0-1
        self.W1 = randomise(self.NI, self.NH)
        self.W2 = randomise(self.NH, self.NO)
        # initialise deltas
        self.delta_hidden = [0.0] * self.NH
        self.delta_output = [0.0] * self.NO

    # forward pass method, takes an input vector and returns output
    # using sigmoid function as default, using tanh function when sig=False
    def forward(self, input, sig=True):
        # activate input layer
        self.I[:(self.NI - 1)] = input
        # activate hidden layer
        for j in range(self.NH):
            activation = 0.0
            for i in range(self.NI):
                activation += self.W1[i][j] * self.I[i]
            if sig:
                self.H[j] = sigmoid(activation)
            else:
                self.H[j] = tanh(activation)
        # activate output layer
        for j in range(self.NO):
            activation = 0.0
            for i in range(self.NH):
                activation += self.W2[i][j] * self.H[i]
            if sig:
                self.O[j] = sigmoid(activation)
            else:
                self.O[j] = tanh(activation)
        return self.O

    # back propagation method for backward pass
    # takes target, learning rate as input and returns error as output
    def backwards(self, target, learningRate, sig=True):
        # get error of output layer
        self.delta_output = [0.0] * self.NO
        for o in range(self.NO):
            error = target[o] - self.O[o]
            if sig:
                self.delta_output[o] = sigmoid_derivative(self.O[o]) * error
            else:
                self.delta_output[o] = tanh_derivative(self.O[o]) * error
        # get error of hidden layer
        self.delta_hidden = [0.0] * self.NH
        for h in range(self.NH):
            error = 0.0
            for o in range(self.NO):
                error += self.delta_output[o] * self.W2[h][o]
            if sig:
                self.delta_hidden[h] = sigmoid_derivative(self.H[h]) * error
            else:
                self.delta_hidden[h] = tanh_derivative(self.H[h]) * error
        # update input and output weights
        self.updateWeights(learningRate)

        # get global error
        error = 0.0
        for o in range(len(target)):
            error += 0.5 * (target[o] - self.O[o]) ** 2
        return error

    # method for updating weights of input and output
    def updateWeights(self, learningRate):
        # weights for input w1
        for m in range(self.NI):
            for n in range(self.NH):
                dw1 = self.delta_hidden[n] * self.I[m]
                self.W1[m][n] += learningRate * dw1
                dw1 = 0

        # weights for output w2
        for i in range(self.NH):
            for j in range(self.NO):
                dw2 = self.delta_output[j] * self.H[i]
                self.W2[i][j] += learningRate * dw2
                dw2 = 0

    # training method which takes examples, targets, number of epochs and learning rates as input
    # calls forward and backwards methods and returns global error
    def train(self, examples, targets, maxEpochs, learningRate, sig=True):
        error_total = []
        for j in range(maxEpochs):
            error = 0.0
            for i in range(len(examples)):
                target = targets[i]
                example = examples[i]
                if sig:
                    self.forward(example)
                    error += self.backwards(target, learningRate)
                else:
                    self.forward(example, False)
                    error += self.backwards(target, learningRate, False)
            error_total.append(error)
        return error_total

if __name__ == '__main__':
    # XOR test
    xor = MLP()
    examples = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    Expects = [[0], [1], [1], [0]]
    xor.initialise(2, 2, 1)
    xor_error = xor.train(examples, Expects, 10000, 0.05)
    # write the error result into csv file
    with open('xor_output.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        f_csv.writerow(['Epoch','Error'])
        for i in range(len(xor_error)):
            row = [i, xor_error[i]]
            f_csv.writerow(row)
    f.close()
    # print out the result
    print('XOR Test:')
    print('Error at last epoch: {}'.format(xor_error[-1]))
    for i in range(len(examples)):
        xor_output = xor.forward(examples[i])
        print('Expect Output: {}, Actual Output: {}'.format(Expects[i], xor_output))

    # 50 sine vectors test
    random.seed(2)
    # generate a 4*50 matrix with components between -1 and 1 as input
    v = [[(2 * random.random() - 1) for i in range(4)] for j in range(50)]
    # get the sine value as output
    out = []
    for row in v:
        out.append([math.sin(row[0] - row[1] + row[2] - row[3])])
    sinvec = MLP()
    sinvec.initialise(4, 5, 1)
    sin_error = sinvec.train(v[:40], out[:40], 10000, 0.05, False)
    # write the error result into csv file
    with open('sinvec_output.csv', 'w') as f:
        f_csv = csv.writer(f, lineterminator='\n')
        f_csv.writerow(['Epoch','Error'])
        for i in range(len(sin_error)):
            row = [i, sin_error[i]]
            f_csv.writerow(row)
    f.close()
    # print out the result
    print('50 Sin Vector Test:')
    print('Error at last epoch: {}'.format(sin_error[-1]))
    for i in range(10):
        sin_output = sinvec.forward(v[-10 + i], False)
        print('Expect Output: {}, Actual Output: {}'.format(out[-10 + i], sin_output))

    # letter recognition
    letter_labels = []
    letter_inputs = []
    # read data from .data file
    f = open('letter-recognition.data', 'r')
    for l in f.readlines():
        l = l.strip()
        content = l.split(',')
        temp = [0.0] * 26
        # get the letter index
        index = lettertoindex(content[0])
        # set the value of the index as 1
        temp[index] = 1.0
        letter_labels.append(temp)
        # get the 16 values as input vector
        letter_inputs.append([int(c) for c in content[1:]])
    f.close()

    lr = MLP()
    lr.initialise(16, 10, 26)
    lr_error = lr.train(letter_inputs[:16000],letter_labels[:16000],1000,0.05)
    # write the error result into csv file
    with open('letter_output.csv', 'w') as f:
            f_csv = csv.writer(f, lineterminator='\n')
            f_csv.writerow(['Epoch','Error'])
            for i in range(1000):
                row = [i, lr_error[i]]
                f_csv.writerow(row)
    f.close()
    # print out the result
    print('Letter Recognition Test:')
    print('Error at last epoch: {}'.format(lr_error[-1]))
    # show the last ten test cases
    for i in range(10):
        lr_output = lr.forward(letter_inputs[-10+i])
        # use method to convert vector to letters
        print('Expect Letter: {}, Actual Output: {}'.format(vectoletter(letter_labels[-10 + i]), lr_output))
        print('Output Letter: {}'.format(vectoletter(lr_output)))


import random as r
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt




class NN():
  def __init__(self):
    self.NNshape = []
    self.inputs = []
    self.W = []
    self.B = []
    self.error = []
    self.hist = []

  def layer(self, a):
    # initializing layer
    self.NNshape.append(a)
  
  def init_hiperparameters(self):
    # initializing weights
    for i in range(len(self.NNshape) - 1):
      self.W.append(np.random.rand(self.NNshape[i+1], self.NNshape[i]))

    # initializing biases
    for j in range(len(self.NNshape) - 1):
      self.B.append(np.random.rand(self.NNshape[j+1], 1))

  def Tanh(Z):
    return(np.tanh(Z))

  def DTanh(Z):
    return 1 - Z**2

  def forward(self, X):
    # preprocess inputs
    I = np.array(X, ndmin=2).T

    # saving first inputs
    self.inputs.append(I)

    for W, B in zip(self.W, self.B):
      # dot product
      X = np.dot(W, I)

      # adding bias
      X = X + B

      # acivation funtion
      I = np.tanh(X)

      # saving next inputs
      self.inputs.append(I)

    return I

  def backprop(self, Y_true, Y_pred, LR):
    ERR = []

    # calculating errors
    E = Y_true - Y_pred
    ERR.append(E)

    for j in range(len(self.W) - 1):
      ERR.append(np.dot(self.W[-j-1].T, ERR[-1]))

    # backpropagation
    for i in range(len(self.W)):
      self.B[-i-1] += LR * ERR[i]
      D = ERR[i] * (1 - self.inputs[-i-1]**2)
      self.W[-i-1] += LR * np.dot(D, self.inputs[-i-2].T)

    return ERR[0]

  def abs_list(self, l):
    k = []
    for i in l:
      k.append(abs(i))
    return k
  
  def show_learning_parameters(self, err, epoch):
    self.error += sum(self.abs_list(err))
    if epoch % 100 == 0:
      self.hist.append(self.error/100)
      self.error = 0
      print('Epoch:', epoch, 'Loss:', self.hist[-1])
    return self.hist

  def early_stopping(self, history, learning_treshold, err_period_low):
    if max(history[len(history) - err_period_low : len(history)]) < learning_treshold:
      return False
    return True
  
  def train(self, dataX, dataY, epochs, learning_rate, early_stopping, *arg):
    ES_params = arg
    train = True
    epoch = 0
    while epoch < epochs and train:
      ind = r.randint(0, len(dataX)-1)
      out = self.forward(dataX[ind])
      E = self.backprop(dataY[ind], out, learning_rate)
      hist = self.show_learning_parameters(E, epoch)
      if early_stopping == True:
        train = self.early_stopping(hist, ES_params[0], ES_params[1])
      epoch += 1
    del(hist[0])
    plt.plot(hist, label=['Loss'])
    plt.legend()
    plt.show()
    return hist


if '__name__' !='__main__':
  NN = NN()
  NN.layer(2)
  NN.layer(5)
  NN.layer(1)
  NN.init_hiperparameters()
  
  data = [[0, 1], [1, 0], [0, 0], [1, 1]]
  targets = [1, 1, 0, 0]
  
  NN.train(data, targets, 500000, 0.003, True, 0.03, 50)
  
  print(NN.forward(data[0]))
  print(NN.forward(data[1]))
  print(NN.forward(data[2]))
  print(NN.forward(data[3]))

Now we will be connecting different Neurons to make up a Neural Network
as i made one in [[Nueron]]

Now there could a lot of hidden layers of Neurons making up a neuron network,
any no of layers between the first and the last is a Hidden layer
what makes it a network is the fact that, input for o1, its gonna be the hiden layer input h1 and h2, 

```
import numpy as np

  

def sigmoid(x):

    return 1/(1 + np.exp(-x))

  

class Neuron:

    def __init__(self, weights,  bias):

        self.weights = weights

        self.bias = bias

  

    def feedforward(self, inputs):

        total = np.dot(self.weights, inputs) + self.bias

        return sigmoid(total)

  

class OurNeuralNetwork:

  '''

  A neural network with:

    - 2 inputs

    - a hidden layer with 2 neurons (h1, h2)

    - an output layer with 1 neuron (o1)

  Each neuron has the same weights and bias:

    - w = [0, 1]

    - b = 0

  '''

  def __init__(self):

    weights = np.array([0, 1])

    bias = 0

  

    # The Neuron class here is from the previous section

    self.h1 = Neuron(weights, bias)

    self.h2 = Neuron(weights, bias)

    self.o1 = Neuron(weights, bias)

  

  def feedforward(self, x):

    out_h1 = self.h1.feedforward(x)

    out_h2 = self.h2.feedforward(x)

  

    # The inputs for o1 are the outputs from h1 and h2

    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

  

    return out_o1

  

network = OurNeuralNetwork()

x = np.array([2, 3])

print(network.feedforward(x))
```
Now what we gonna do here, it create a new class , and pass the two inputs with the same weights and bias, but we will see since , their output is an input to the o1, output which is another neuron and then it will output again, revealing a diff value, than the one from before
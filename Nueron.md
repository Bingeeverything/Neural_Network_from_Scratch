So basically what i learned is that a neuron is a single node of the thing that processes things in a AI, network. and if we stack these things up, we get a neural network.


So what this is basicallly is , there is When processing somethin, we take in input and output it, in a single one, takes the input, does the math with it and returns the output

The math here is just, asigning the numbers with some bias and using a sigmoid function, which gives us a number, and acc to the number, we do what we do, its just a probability, and if more positive, it does that,

Now, why Sigmoid, so the thing is that, its sigmoid takes in all sets of no, all from  negative infinity to positive infinity, same as any other, but why we use it is because it compesses the value to between 0 and 1.

MATHS-

Now on to the maths side, 
lets say we have 2 inputs, we first multiply it with some weight, so 
$$
x1​→x1​∗w1
$$
$$
​x2→x2∗w2
$$

add it with a bias b
$$
(x1​∗w1​)+(x2​∗w2​)+b
$$
and pass it through a activation function, which is out sigmoid in this case
$$
y=f(x1​∗w1​+x2​∗w2​+b)
$$

import numpy as np

  

```
def sigmoid(x):

    return 1/(1 + np.exp(-x))

  

class Neuron:

    def __init__(self, weights,  bias):

        self.weights = weights

        self.bias = bias

  

    def feedforward(self, inputs):

        total = np.dot(self.weights, inputs) + self.bias

        return sigmoid(total)

weights = np.array([0,1])

bias = 3

n = Neuron(weights, bias)

  

x = np.array([2,3])

print(n.feedforward(x))
```
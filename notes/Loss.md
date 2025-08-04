No we will combine what we learnt in both [[Nueron]] [[Combining Networks]] and try to train a basic Neural network

| Name  | Weight | height | Gender |
| ----- | ------ | ------ | ------ |
| Alice | 55     | 157    | F      |
| Bob   | 67     | 170    | B      |
| Bonku | 73     | 167    | F      |
| Rey   | 58     | 159    | B      |
Now will train our network to guess if its a male or a female based on weight and height

making our F 1 and M 0

No before training our network, what we gonna do is, calculate loss
which tells our network, if its not good, the result
it can try to be good
Thats why we first need to tell what 'good' is, so that it can try to do 'better'

That is LOSS

$$
MSE==1/n∑n​(ytrue​−ypred​)2
$$
here,
n is the size of the sample, so 4
y represents the thing being predicted (gender)
y true is the real correct value, like F for alice]
y pred is the output 

$$
(ytrue​−ypred​)2
$$
 this is just squared mean
 so were just calculating the average, because the better our predictions, the lower our loss

Lets say our network predicts all the people are male, so 0
our MSE(Mean squared error) would be 0.5
```
import numpy as np

  

def mse_loss(y_true, y_pred):

  # y_true and y_pred are numpy arrays of the same length.

  return ((y_true - y_pred) ** 2).mean()

  

y_true = np.array([1, 0, 0, 1])

y_pred = np.array([0, 0, 0, 0])

  

print(mse_loss(y_true, y_pred))
```

this outputs 0.5
# Network From Scratch

This project is my implementation of a neural network in Java. everything has been implemented by me,
from matrix multiplication to backpropagation.

## Setup

to get the project up and running simply use:
- ```.\gradlew build```

to run the given example:
- ```.\gradlew run```

## Example
Here we create our neural network object and set up our dataset
```
NeuralNetwork nn = new NeuralNetwork()

DataSet dataSet = DataSet.split(ReadIO.readCsv("BostonHousing.csv"), 1);

NDArray x = dataSet.getX();
NDArray y = dataSet.getY();
```

Add a layer of size 40. The input of a layer must be the same size as the output of the previous, which in this case is our data's feature size.
the last paramater is an activation function. in this case its a Leaky ReLU function.
```
nn.layer(new Layer(x.shape[1], 40, new LeakyReLU()))

```

Create some more layers.
```
nn.layer(new Layer(100, 10, new LeakyReLU()))
  .layer(new Layer(10, 10, new LeakyReLU()))
  .layer(new Layer(10, y.shape()[1], new LeakyReLU()));
```

Define an appropriate learning rate.
```
nn.learningRate(0.00001);
```

If we would like to see the progress of the cost function:
```
nn.plotCostGraph(true)
```

Train the network
```
int epochs = 1000
int batchSize = 50

nn.train(dataSet, epochs, batchSize);
```

Predict the labels using the trained network.
```
NDArray result = nn.predict(x);
```





## Valuable sources

Check out [this lecture](https://www.youtube.com/watch?v=Ixl3nykKG9M&t=6045s), which tells you all about the maths behind neural networks.

the lecture is inspired by these books:
- [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528)
- [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)



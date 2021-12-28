# Network From Scratch

This project is my implementation of a neural network in Java. everything has been implemented by me,
from matrix multiplication to backpropagation.

## Setup

to get the project up and running simply use:
- ```.\gradlew build```

to run the given example:
- ```.\gradlew run```

## Example
 here we create our neural network object with an input size of 2
```
NeuralNetwork nn = new NeuralNetwork(2)
```

Add a layer which has the same input size as the network
the last paramater is an activation function. in this case its a ReLU function
```
nn.layer(new Layer(2, 4, new ReLU()))

```

Add another layer with the inputsize of the previous layer and an output size of 1.
This layer uses Sigmoid for its activation function.
```
nn.layer(new Layer(4, 1, new Sigmoid()))
```

We set the learning rate to 0.01
```
nn.learningRate(0.01)
```

If we would like to see the progress of the cost function:
```
nn.plotCostGraph(true)
```

Define our training data using an NDArray object
```
        NDArray xTrain = new NDArray(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        });
        NDArray yTrain = new NDArray(new double[][]{
                {0},
                {0},
                {0},
                {1},
        }); 
```

Train the network using the following data for 5000 epochs
```
nn.train(xTrain, yTrain, 5000);
```

Predict the labels using the trained network
```
NDArray result = nn.predict(xTrain);
```





## Valuable sources

Check out [this lecture](https://www.youtube.com/watch?v=Ixl3nykKG9M&list=WL&index=2), which tells you all about the maths behind neural networks.

the lecture is inspired by these books:
- [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528)
- [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)



package network.neural;

import network.neural.activationfunctions.IActivationFunction;

/**
 * Simple 1 layer network.
 */
public class Network {

    private NDArray weights;
    private NDArray hiddenWeights;
    private int numInputs;

    private IActivationFunction activationFunction;

    private NDArray xTrain;
    private NDArray yTrain;

    public Network(int numInputs, int hiddenSize, int numOutputs, IActivationFunction activationFunction) {
        this.numInputs = numInputs;
        this.activationFunction = activationFunction;
//        this.hiddenWeights = NDArray.rand(hiddenSize, numInputs + 1);
//        this.weights = NDArray.rand(numOutputs, hiddenSize + 1);
        this.weights = NDArray.zeros(numOutputs, numInputs + 1);
    }

    public void train(NDArray xTrain, NDArray yTrain) {
        if (xTrain.shape()[1] != numInputs)
            throw new IllegalArgumentException("Expected " + numInputs + " inputs.");

        if (xTrain.shape()[0] != yTrain.shape()[0])
            throw new IllegalArgumentException("training set is inconsistent.");

        this.xTrain = xTrain;
        this.yTrain = yTrain;

        for (int i = 0; i < 100000; i++) {
            backpropagation();
        }
    }

    private void backpropagation() {
        NDArray costMatrix = cost(yTrain, feedForward(xTrain));
        NDArray gradient = costMatrix.T().dot(addBiases(xTrain));
        weights = weights.add(gradient.mul(0.01));
    }

    private NDArray cost(NDArray y, NDArray yHat) {
        return y.sub(yHat);
    }


    public NDArray predict(NDArray inputs) {
        return feedForward(inputs);
    }

    private NDArray feedForward(NDArray inputs) {
        inputs = addBiases(inputs);
//        NDArray hidden = inputs.dot(hiddenWeights).activation(activationFunction);
//        System.out.println(hidden);
//        hidden = hiddenWeights.dot(inputs.T()).activation(activationFunction).T();
//        System.out.println(hidden);
//        return weights.dot(inputs.T()).activation(activationFunction).T();
        return weights.dot(inputs.T()).activation(activationFunction).T();
//        return null;
    }

    /**
     * Concatenates a double column of ones to the beginning of an NDArray.
     * @param input
     * @return
     */
    private NDArray addBiases(NDArray input) {
        double[][] result = new double[input.shape()[0]][input.shape()[1] + 1];

        double[][] inputArray = input.data();

        for (int i = 0; i < input.shape()[0]; i++) {
            result[i][0] = 1;
            for (int j = 0; j < input.shape()[1]; j++) {
                result[i][j + 1] = inputArray[i][j];
            }
        }

        return new NDArray(result);
    }

    public NDArray getWeights() {
        return weights;
    }
}

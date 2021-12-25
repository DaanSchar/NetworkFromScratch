package network.neural;

import network.neural.activationfunctions.IActivationFunction;

import java.util.Arrays;

/**
 * Simple 1 layer network.
 */
public class Network {

    private NDArray weights;
    private NDArray hiddenWeights;

    private NDArray biases;
    private NDArray hiddenBiases;

    private int numInputs;

    private IActivationFunction activationFunction;

    private NDArray xTrain;
    private NDArray yTrain;

    private double learningRate = 0.01;

    // output = X^T dot W + b

    public Network(int numInputs, int hiddenSize, int numOutputs, IActivationFunction activationFunction) {
        this.numInputs = numInputs;
        this.activationFunction = activationFunction;

        this.hiddenWeights = NDArray.rand(numInputs, hiddenSize);
        this.weights = NDArray.rand(hiddenSize, numOutputs);

        this.hiddenBiases = NDArray.rand(1, hiddenSize);
        this.biases = NDArray.rand(1, numOutputs);

//        this.weights = NDArray.zeros(numOutputs, numInputs + 1);
    }

//    public void train(NDArray xTrain, NDArray yTrain) {
//        if (xTrain.shape()[1] != numInputs)
//            throw new IllegalArgumentException("Expected " + numInputs + " inputs.");
//
//        if (xTrain.shape()[0] != yTrain.shape()[0])
//            throw new IllegalArgumentException("training set is inconsistent.");
//
//        this.xTrain = xTrain;
//        this.yTrain = yTrain;
//
//        for (int i = 0; i < 100000; i++) {
//            denseLayerBackpropagation();
//            hiddenLayerBackPropagation();
//        }
//    }

    public void train(NDArray X, NDArray Y) {
        NDArray hidden = hiddenWeights.dot(X.T()).add(hiddenBiases).activation(activationFunction);
        NDArray output = weights.dot(hidden).add(biases).activation(activationFunction);
        NDArray error = error(Y, output);
        System.out.println(Arrays.deepToString(error.data()));
        NDArray gradient = output.gradient(activationFunction).mul(error).mul(learningRate);
        NDArray delta = gradient.dot(hidden.T());
        this.weights = weights.add(delta);
        this.biases = biases.add(gradient);

        NDArray hiddenError = weights.T().dot(error);
        NDArray hiddenGradient = hidden.gradient(activationFunction).mul(hiddenError).mul(learningRate);
        NDArray hiddenDelta = hiddenGradient.dot(X);
        this.hiddenWeights = hiddenWeights.add(hiddenDelta);
        this.hiddenBiases = hiddenBiases.add(hiddenGradient);
    }

    public void fit(NDArray X, NDArray Y, int epochs) {
        if (X.shape()[1] != numInputs)
            throw new IllegalArgumentException("Expected " + numInputs + " inputs.");

        if (X.shape()[0] != Y.shape()[0])
            throw new IllegalArgumentException("training set is inconsistent. " + xTrain.shape()[0] + " != " + yTrain.shape()[0]);

        for(int i=0;i<epochs;i++) {
            int sampleN =  (int)(Math.random() * X.shape()[0]);
            this.train(
                    new NDArray(X.data()[sampleN]),
                    new NDArray(Y.data()[sampleN])
            );
        }
    }

    public void derivative(NDArray X, NDArray Y) {
        this.xTrain = X;
        this.yTrain = Y;

        for (int i = 0; i < xTrain.shape()[0]; i++) {
            NDArray a1 = xTrain.getRow(i);
            NDArray z2 = a1.dot(this.hiddenWeights).addVector(this.hiddenBiases);
            NDArray a2 = z2.activation(activationFunction);
            NDArray z3 = a2.dot(this.weights).addVector(this.biases);
            NDArray a3 = z3.activation(activationFunction);
            NDArray d3 = a3.sub(yTrain.getRow(i));
            NDArray delta2 = NDArray.zeros(this.weights.shape());
            delta2.data()[i] = d3.dot(a3.T()).data()[0];
            NDArray d2 = weights.T().mul(d3.data()[0][0]).mul(z2.gradient(activationFunction));
            System.out.println(d2);
            NDArray delta1 = d2.dot(a1.T());
        }


//        NDArray delta3 = error.mul(z3.gradient(activationFunction)).dot(this.weights.T());
//        System.out.println(delta3);
//        NDArray delta2 = delta3.dot(this.weights.T()).mul(z2.gradient(activationFunction));

    }

    public NDArray predict(NDArray inputs) {
        NDArray z2 = hiddenWeights.dot(inputs).addVector(hiddenBiases);
        NDArray a2 = z2.activation(activationFunction);
        NDArray z3 = weights.dot(a2).addVector(biases);
        NDArray a3 = z3.activation(activationFunction);
        return a3;
    }

    public void setWeights(NDArray weights) {
        this.weights = weights;
    }

    public void setBiases(NDArray biases) {
        this.biases = biases;
    }

    public void setHiddenWeights(NDArray hiddenWeights) {
        this.hiddenWeights = hiddenWeights;
    }

    public void setHiddenBiases(NDArray hiddenBiases) {
        this.hiddenBiases = hiddenBiases;
    }

    private NDArray layer1FeedForward(NDArray inputs) {
        inputs = addBiases(inputs);
        return hiddenWeights.dot(inputs.T()).activation(activationFunction).T();
    }

    private NDArray layeredFeedForward(NDArray inputs) {
        return layer2FeedForward(layer1FeedForward(inputs));
    }

    private NDArray layer2FeedForward(NDArray inputs) {
        inputs = addBiases(inputs);
        return weights.dot(inputs).activation(activationFunction).T();
    }

    private void hiddenLayerBackPropagation() {
        NDArray costMatrix = error(yTrain, layeredFeedForward(xTrain));
        costMatrix = addBiases(costMatrix);
        NDArray gradient = hiddenWeights.T().dot(costMatrix.T());

//        NDArray gradient2 = gradient.mul(layer1FeedForward(xTrain).gradient(activationFunction));
//        hiddenWeights = hiddenWeights.add(gradient2.mul(0.01));
    }

    private void denseLayerBackpropagation() {
        NDArray costMatrix = error(yTrain, layeredFeedForward(xTrain));
        NDArray gradient = costMatrix.T().dot(addBiases((layer1FeedForward(xTrain))));
        weights = weights.add(gradient.mul(0.01));
    }

    private NDArray error(NDArray y, NDArray yHat) {
        return y.sub(yHat);
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

package network.neural.old;

import network.neural.activationfunctions.IActivationFunction;
import network.neural.charts.LineChart;
import network.neural.util.NDArray;


/**
 * "Simple" 2 layer network.
 *
 * I'm leaving this class for later use.
 */
public class TwoLayerNetwork {

    private NDArray weights;
    private NDArray hiddenWeights;

    private NDArray biases;
    private NDArray hiddenBiases;

    private int numInputs;

    private IActivationFunction activationFunction;

    private NDArray xTrain;
    private NDArray yTrain;

    private double learningRate = 0.01;

    public TwoLayerNetwork(int numInputs, int hiddenSize, int numOutputs, IActivationFunction activationFunction) {
        this.numInputs = numInputs;
        this.activationFunction = activationFunction;

        double epsilon = Math.sqrt(6) / (numInputs + hiddenSize);

        this.hiddenWeights = NDArray.rand(epsilon, hiddenSize, numInputs);
        this.weights = NDArray.rand(epsilon, numOutputs, hiddenSize);

        this.hiddenBiases = NDArray.rand(hiddenSize,1);
        this.biases = NDArray.rand(numOutputs, 1);
    }


    /**
     * Trains the network using forward and backward propagation.
     *
     * @param X training data
     * @param Y labels
     */
    public void train(NDArray X, NDArray Y, int epochs) {
        this.xTrain = X;
        this.yTrain = Y;

        double[] costs = new double[epochs];

        for (int i = 0; i < epochs; i++) {
            costs[i] = backpropagation();
        }

        // plot the cost
        LineChart chart = new LineChart(
                costs,
                "Average Cost",
                "Epoch",
                "Cost"
        );
        chart.setVisible(true);
    }

    /**
     * Predict the output of the network by using forward propagation.
     *
     * @param inputs input data we want to make a prediction on
     * @return the prediction
     */
    public NDArray predict(NDArray inputs) {
        NDArray z2 = hiddenWeights.dot(inputs.T()).addVector(hiddenBiases).T();
        NDArray a2 = z2.activation(activationFunction);
        NDArray z3 = weights.dot(a2.T()).addVector(biases).T();
        NDArray a3 = z3.activation(activationFunction);
        return a3;
    }


    /**
     * Backpropagation algorithm.
     *
     * derivative of a FinalLayer L = derivative(cost) * activation'(z^L)
     * derivative of any other layer l = (W^l+1 dot derivative^l+1) * activation'(z^l)
     *
     * bias^l = bias^l + derivative^l
     * weights^l = weights^l + derivative^l dot activation^l-1
     *
     * @return cost of the current state of the weights and biases on the
     *         training data.
     */
    private double backpropagation() {
        // Forward propagation
        NDArray z2 = hiddenWeights.dot(xTrain.T()).addVector(hiddenBiases).T();
        NDArray a2 = z2.activation(activationFunction);
        NDArray z3 = weights.dot(a2.T()).addVector(biases).T();
        NDArray a3 = z3.activation(activationFunction);

        // last layer error = cost'(a) * activation'(z)
        NDArray errorLastLayer = costDerivative(yTrain, a3)
                .mul(
                        z3.gradient(activationFunction)
                ).T();

        // hidden layer error = W^T dot errorLastLayer * activation'(z)^T
        NDArray errorHiddenLayer = weights.T().dot(errorLastLayer).mul(z2.gradient(activationFunction).T());

        // new biases = old biases + alpha*error
        this.biases = biases.add(errorLastLayer.mul(learningRate).getAvgColVector());
        this.hiddenBiases = hiddenBiases.add(errorHiddenLayer.mul(learningRate).getAvgColVector());

        // new weights = old weights + alpha*error dot (activation of previous layer)
        this.weights = weights.add(errorLastLayer.dot(a2).mul(learningRate));
        this.hiddenWeights = hiddenWeights.add(errorHiddenLayer.dot(xTrain).mul(learningRate));

        return (cost(yTrain, a3).sum());
    }


    /**
     * Cost function.
     *
     * @param y labels
     * @param yHat predictions
     * @return error squared vector
     */
    private NDArray cost(NDArray y, NDArray yHat) {
        return y.sub(yHat).pow(2);
    }


    /**
     * Derivative of the cost function.
     *
     * @param y labels
     * @param yHat predictions
     * @return error' vector
     */
    private NDArray costDerivative(NDArray y, NDArray yHat) {
        return y.sub(yHat).mul(2);
    }

}

package network.neural.layers;

import network.neural.LayerOutput;
import network.neural.MultiNetwork;
import network.neural.NDArray;
import network.neural.activationfunctions.IActivationFunction;

public class Layer {

    private int inputSize;
    private int outputSize;
    private NDArray error;

    private NDArray weights;
    private NDArray bias;

    private MultiNetwork network;

    private IActivationFunction activationFunction;

    /**
     * layer of neurons with weights and biases.
     *
     * @param inputSize number of nodes going in to the layer
     * @param outputSize number of nodes going out of the layer, or the number of
     *                   neurons in the layer
     *
     * @param activationFunction the activation function to use in the layer
     */
    public Layer(int inputSize, int outputSize, IActivationFunction activationFunction) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationFunction = activationFunction;

        double epsilon = Math.sqrt(6) / Math.sqrt(inputSize + outputSize);
        this.weights = NDArray.rand(epsilon, outputSize, inputSize);
        this.bias = NDArray.rand(epsilon,  outputSize, 1);
    }

    /**
     * Predict the output of the network by using forward propagation.
     *
     * @param input input data we want to make a prediction on
     * @return array where [0] is the output z and [1] is the activation a
     */
    public LayerOutput forward(NDArray input) {
        NDArray z = weights.dot(input.T()).addVector(bias).T();
        NDArray a = z.activation(activationFunction);

        return new LayerOutput(z, a);
    }

    public void update(NDArray error, double learningRate) {
        updateWeights(error);
        updateBias(error);
    }

    protected void updateWeights(NDArray error) {
//        weights = weights.add(error)
    }

    protected void updateBias(NDArray error) {

    }

    public NDArray getError(NDArray errorNextLayer, LayerOutput outputThisLayer) {
        return weights.T().dot(errorNextLayer);//.mul(outputThisLayer.getZ().gradient(activationFunction).T());
    }

    public void setNetwork(MultiNetwork network) {
        this.network = network;
    }

    public MultiNetwork getNetwork() {
        return network;
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public void setWeights(NDArray weights) {
        this.weights = weights;
    }

    public NDArray getWeights() {
        return weights;
    }

    public void setBias(NDArray bias) {
        this.bias = bias;
    }

    public NDArray getBias() {
        return bias;
    }

    public IActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public String toString() {
        return "Layer{ inputSize: " + inputSize + ", outputSize: " + outputSize + ", activation " + activationFunction + "}";
    }

    public void setError(NDArray error) {
        this.error = error;
    }

    public NDArray getError() {
        return error;
    }
}

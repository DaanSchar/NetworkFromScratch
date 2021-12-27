package network.neural.layers;

import network.neural.LayerOutput;
import network.neural.NDArray;
import network.neural.activationfunctions.IActivationFunction;

public class FinalLayer extends Layer {
    /**
     * layer of neurons with weights and biases.
     *
     * @param inputSize          number of nodes going in to the layer
     * @param outputSize         number of nodes going out of the layer, or the number of
     *                           neurons in the layer
     * @param activationFunction the activation function to use in the layer
     */
    public FinalLayer(int inputSize, int outputSize, IActivationFunction activationFunction) {
        super(inputSize, outputSize, activationFunction);
    }

    public NDArray getError(LayerOutput output, NDArray labels) {
        return costDerivative(labels, output.getA())
                .mul(
                        output.getZ().gradient(super.getActivationFunction())
                ).T();
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


    protected void updateWeights(NDArray error) {
        super.setBias(super.getBias().add(error.mul(super.getNetwork().getLearningRate()).getAvgColVector()));
    }

    protected void updateBias() {

    }

}

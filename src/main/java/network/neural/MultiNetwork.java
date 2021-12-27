package network.neural;

import network.neural.layers.FinalLayer;
import network.neural.layers.Layer;

import java.util.ArrayList;

public class MultiNetwork {

    private final ArrayList<Layer> layers;
    private final int inputSize;

    private NDArray xTrain;
    private NDArray yTrain;

    private double learningRate;

    public MultiNetwork(int inputSize) {
        this.inputSize = inputSize;
        layers = new ArrayList<>();
    }

    /**
     * adds a layer to the network
     *
     * @return the network
     */
    public MultiNetwork layer(Layer layer) {
        layer.setNetwork(this);
        layers.add(layer);
        return this;
    }

    /**
     * Predict the output of the network by using forward propagation.
     *
     * @param input data we want to make a prediction on
     * @return the output of the last layer.
     */
    public NDArray predict(NDArray input) {
        ArrayList<LayerOutput> outputs = forward(input);

        return outputs.get(outputs.size() - 1).getA();
    }

    public void train(NDArray x, NDArray y, int epochs) {
        this.xTrain = x;
        this.yTrain = y;
        double error = 0;
        double[] errors = new double[epochs];

        for (int i = 0; i < epochs; i++) {
            errors[i] = backpropagate();
        }

        CostChart chart = new CostChart(errors, "Cost", "Epochs", "Cost");
        chart.setVisible(true);
    }

    private NDArray getFinalError(LayerOutput lastOutput) {
        FinalLayer finalLayer = (FinalLayer) layers.get(layers.size() - 1);
        return  finalLayer.getError(lastOutput, yTrain);
    }

    private void updateFinalLayer(NDArray errorLast, LayerOutput prevLayerOutput) {
        FinalLayer finalLayer = (FinalLayer) layers.get(layers.size() - 1);

        NDArray newFinalWeights = finalLayer.getWeights().add(errorLast.dot(prevLayerOutput.getA()).mul(learningRate));
        finalLayer.setWeights(newFinalWeights);
        finalLayer.setBias(finalLayer.getBias().add(errorLast.mul(learningRate).getAvgColVector()));
    }

    private double backpropagate() {
        ArrayList<LayerOutput> outputs = forward(xTrain);

        NDArray errorLast = getFinalError(outputs.get(outputs.size() - 1));
        layers.get(layers.size() - 1).setError(errorLast);

        for  (int i = layers.size()-2; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);

            NDArray error = nextLayer
                    .getWeights().T()
                    .dot(nextLayer.getError())
                    .mul(
                            outputs.get(i+1)
                                    .getZ()
                                    .gradient(currentLayer.getActivationFunction()).T()
                    );
            layers.get(i).setError(error);
        }

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);

            NDArray newWeights = layer.getWeights().add(
                    layer.getError().dot(outputs.get(i).getA())
                            .mul(learningRate)
            );
            NDArray newBias = layer.getBias().add(layer.getError().mul(learningRate).getAvgColVector());
            layer.setWeights(newWeights);
            layer.setBias(newBias);
        }



//        Layer currentLayer = layers.get(layers.size() - 2);
//        Layer nextLayer = layers.get(layers.size() - 1);
//        NDArray error = nextLayer
//                .getWeights().T()
//                .dot(errorLast)
//                .mul(
//                        outputs.get(outputs.size() - 2).getZ()
//                                .gradient(currentLayer.getActivationFunction()
//                                ).T()
//                );

        return cost(yTrain, outputs.get(outputs.size() - 1).getA()).sum();
    }

    /**
     * does forward propagation on each layer of the network
     * and returns the output of each layer
     *
     * @param input data we want to make a prediction on
     * @return the prediction
     */
    public ArrayList<LayerOutput> forward(NDArray input) {
        ArrayList<LayerOutput> outputs = new ArrayList<>();
        outputs.add(new LayerOutput(input, input));

        for (Layer layer : layers)
            outputs.add(
                    layer.forward(outputs.get(outputs.size() - 1).getA())
            );

        return outputs;
    }


    /**
     * Cost function.
     *
     * @param y labels
     * @param yHat predictions
     * @return error squared vector
     */
    public NDArray cost(NDArray y, NDArray yHat) {
        return y.sub(yHat).pow(2);
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getLayerCount() {
        return layers.size();
    }

    public Layer getLayer(int index) {
        return layers.get(index);
    }

    public double getLearningRate() {
        return learningRate;
    }

    public MultiNetwork learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
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

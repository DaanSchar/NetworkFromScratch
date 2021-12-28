package network.neural;

import network.neural.charts.LineChart;
import network.neural.layer.Layer;
import network.neural.layer.LayerOutput;
import network.neural.util.matrix.NDArray;
import network.neural.util.readwrite.ObjectIO;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * A multi-layer, fully connected, artificial neural network.
 * support for arbitrary number of layers with an arbitrary number of neurons.
 */
public class NeuralNetwork implements Serializable {

    private final ArrayList<Layer> layers;
    private final int inputSize;

    private double learningRate;
    private boolean costGraph;

    private NDArray xTrain;
    private NDArray yTrain;


    public NeuralNetwork(int inputSize) {
        this.inputSize = inputSize;
        layers = new ArrayList<>();
    }


    /**
     * adds a layer to the network
     *
     * @return the network
     */
    public NeuralNetwork layer(Layer layer) {
        layer.setNetwork(this);
        layers.add(layer);
        return this;
    }


    /**
     * sets the learning rate of the network
     *
     * @param learningRate the learning rate
     * @return the network
     */
    public NeuralNetwork learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }


    /**
     * boolean if you want to see the graph of the cost
     */
    public NeuralNetwork plotCostGraph(boolean graph) {
        this.costGraph = graph;
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

    /**
     * trains the network on the given dataset using backpropagation
     *
     * @param x the input data
     * @param y the labels for the input data
     * @param epochs the number of epochs to train for
     */
    public void train(NDArray x, NDArray y, int epochs) {
        this.xTrain = x;
        this.yTrain = y;
        double[] errors = new double[epochs];

        for (int i = 0; i < epochs; i++)
            errors[i] = backpropagation();

        if (costGraph)
            createCostGraph(errors);
    }


    /**
     * saves the state of the network to a file
     *
     * @param path location of the file
     */
    public void save(String path) {
        ObjectIO.WriteObjectToFile(this, path);
    }


    /**
     * load a neural network from a file.
     *
     * @param path location of the file
     * @return the neural network
     */
    public static NeuralNetwork load(String path) {
        return (NeuralNetwork) ObjectIO.readObjectFromFile(path);
    }


    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return layers.get(layers.size() - 1).getOutputSize();
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


    /**
     * performs one time backward propagation
     *
     * @return the cost
     */
    private double backpropagation() {
        ArrayList<LayerOutput> outputs = forward(xTrain);

        updateFinalLayerError(outputs.get(outputs.size() - 1));
        updateLayerError(outputs);
        updateLayerWeights(outputs);

        return cost(yTrain, outputs.get(outputs.size() - 1).getA()).sum();
    }


    /**
     * Computes the error of the final layer.
     * error of a FinalLayer L = derivative(cost) * activation'(z^L)
     *
     * @param lastOutput the output of the last layer
     */
    private void updateFinalLayerError(LayerOutput lastOutput) {
        Layer finalLayer = layers.get(layers.size() - 1);
        NDArray error = costDerivative(yTrain, lastOutput.getA()).mul(lastOutput.getZ().gradient(finalLayer.getActivationFunction())).T();
        finalLayer.setError(error);
    }


    /**
     * updates the error of each hidden layer
     * error of a layer l = (W^l+1 dot derivative^l+1) * activation'(z^l)
     *
     * @param outputs list of forward propagation outputs of all the layers layer
     */
    private void updateLayerError(ArrayList<LayerOutput> outputs) {
        for  (int i = layers.size()-2; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);

            NDArray error = nextLayer
                    .getWeights().T()
                    .dot(nextLayer.getError())
                    .mul(outputs.get(i+1).getZ().gradient(currentLayer.getActivationFunction()).T());
            layers.get(i).setError(error);
        }
    }


    /**
     * updates the weights of each layer
     * using the error of the layer computed in updateLayerError().
     * weights^l += error^l dot activation^l-1
     *
     * @param outputs list of forward propagation outputs of all the layers layer
     */
    private void updateLayerWeights(ArrayList<LayerOutput> outputs) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);

            NDArray newWeights = layer.getWeights().add(layer.getError().dot(outputs.get(i).getA()).mul(learningRate));
            NDArray newBias = layer.getBias().add(layer.getError().mul(learningRate).getAvgColVector());

            layer.setWeights(newWeights);
            layer.setBias(newBias);
        }
    }


    /**
     * does forward propagation on each layer of the network
     * and returns the output of each layer
     *
     * @param input data we want to make a prediction on
     * @return the prediction
     */
    private ArrayList<LayerOutput> forward(NDArray input) {
        ArrayList<LayerOutput> outputs = new ArrayList<>();
        outputs.add(new LayerOutput(input, input));

        for (Layer layer : layers)
            outputs.add(layer.forward(outputs.get(outputs.size() - 1).getA()));

        return outputs;
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


    /**
     * Creates a graph showing the cost of the network for each epoch
     *
     * @param errors list of errors for each epoch
     */
    private void createCostGraph(double[] errors) {
        LineChart lineChart = new LineChart(errors, "Cost", "Epochs", "Cost");
        lineChart.create();
        lineChart.setVisible(true);
    }



}

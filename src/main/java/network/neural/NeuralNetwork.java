package network.neural;

import me.tongfei.progressbar.ProgressBar;
import network.neural.charts.LineChart;
import network.neural.layer.Layer;
import network.neural.layer.LayerOutput;
import network.neural.util.DataSet;
import network.neural.util.matrix.NDArray;
import network.neural.util.readwrite.ObjectIO;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

/**
 * A multi-layer, fully connected, artificial neural network.
 * support for arbitrary number of layers with an arbitrary number of neurons.
 */
public class NeuralNetwork implements Serializable {

    private final ArrayList<Layer> layers;

    private double learningRate;
    private boolean costGraph;

    private DataSet trainingSet;
    private DataSet[] batchList;
    private int m;

    public NeuralNetwork() {
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
     * @param trainingSet the dataset containing x train and y train
     * @param epochs the number of epochs to train for
     */
    public void train(DataSet trainingSet, int epochs, int batchSize) {
        this.trainingSet = trainingSet;
        this.batchList = DataSet.batch(trainingSet, batchSize);
        this.m = batchList.length * batchSize;

        double[] errors = train(epochs);

        if (costGraph)
            createCostGraph(errors);
    }

    /**
     * trains the network, but wraps it in a progress bar
     * @return the cost for each epoch
     */
    private double[] train(int epochs) {
        double[] costs = new double[epochs];
        long start = System.currentTimeMillis();

//        try (ProgressBar pb = new ProgressBar("Training", epochs)) {
//            for (int i = 0; i < epochs; i++) {
//                for (int j = 0; j < batchList.length; j++) {
//                    costs[i] += backpropagation(j);
//                }
//                pb.step();
//            }
//        }
        try (ProgressBar pb = new ProgressBar("Training", epochs)) {
            for (int epoch = 0; epoch < epochs; epoch++) {

                BackPropThread[] threads = new BackPropThread[batchList.length];

                for (int i = 0; i < batchList.length; i++) {
                    threads[i] = new BackPropThread(i);
                    threads[i].start();
                }

                try {
                    for (BackPropThread thread : threads)
                        thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                NDArray[] errors = threads[0].getErrors();

                for (int i = 0; i < errors.length; i++) {
                    for (int j = 1; j < threads.length; j++) {
                        errors[i] = errors[i].add(threads[j].getErrors()[i]);
                    }
                    layers.get(i).setError(errors[i].mul(1.0 / (double) batchList.length));
                }

//                updateWeights(forward(batchList[new Random().nextInt(batchList.length-1)].getX()));

                for (int i = 0; i < batchList.length; i++) {
                    ArrayList<LayerOutput> outputs = forward(batchList[i].getX());
                    updateWeights(forward(batchList[i].getX()));
                    double cost = cost(batchList[i].getY(), outputs.get(outputs.size() - 1).getA()).sum();
                    costs[epoch] += cost;
                }
                pb.step();
            }
        }
        long end = System.currentTimeMillis();
        System.out.println("Time: " + (end - start) / 1000.0);

        return costs;
    }

    private double getCost() {
        double cost = 0;
        for (int i = 0; i < batchList.length; i++) {
            ArrayList<LayerOutput> outputs = forward(batchList[i].getX());
            cost += cost(batchList[i].getY(), outputs.get(outputs.size() - 1).getA()).sum();
        }
        return cost;
    }

    class BackPropThread extends Thread {
        private NDArray[] errors;
        private int batch;

        public BackPropThread(int batch) {
            this.batch = batch;
        }

        @Override
        public void run() {
            errors = backpropagations(batch);
        }

        public NDArray[] getErrors() {
            return errors;
        }
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
    private double backpropagation(int batchIndex) {
        ArrayList<LayerOutput> outputs = forward(batchList[batchIndex].getX());

        NDArray[] errors = getErrors(outputs, batchIndex);

        for (int i = 0; i < layers.size(); i++) {
            layers.get(i).setError(errors[i]);
        }

        updateWeights(outputs);

        return cost(batchList[batchIndex].getY(), outputs.get(outputs.size() - 1).getA()).sum();
    }

    private NDArray[] backpropagations(int batchIndex) {
        ArrayList<LayerOutput> outputs = forward(batchList[batchIndex].getX());

        return getErrors(outputs, batchIndex);
    }


    /**
     * updates the error of each hidden layer
     * error of a layer l = (W^l+1 dot derivative^l+1) * activation'(z^l)
     *
     * @param batchIndex the index of the batch
     * @param outputs list of forward propagation outputs of all the layers layer
     * @return array of errors of each layer
     */
    private NDArray[] getErrors(ArrayList<LayerOutput> outputs, int batchIndex) {
        NDArray[] errors = new NDArray[layers.size()];
        errors[layers.size()-1] = getFinalLayerError(outputs.get(outputs.size() - 1), batchIndex);

        for  (int i = layers.size()-2; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);

            NDArray error = nextLayer
                    .getWeights().T()
                    .dot(errors[i+1])
                    .mul(outputs.get(i+1).getZ().gradient(currentLayer.getActivationFunction()).T());

            errors[i] = error;
        }

        return errors;
    }

    /**
     * Computes the error of the final layer.
     * error of a FinalLayer L = derivative(cost) * activation'(z^L)
     *
     * @param lastOutput the output of the last layer
     * @param batchIndex the index of the batch
     * @return the error of the final layer
     */
    private NDArray getFinalLayerError(LayerOutput lastOutput, int batchIndex) {
        Layer finalLayer = layers.get(layers.size() - 1);
        return costDerivative(batchList[batchIndex].getY(), lastOutput.getA()).mul(lastOutput.getZ().gradient(finalLayer.getActivationFunction())).T();
    }


    /**
     * updates the weights of each layer
     * using the error of the layer computed in updateLayerError().
     * weights^l += error^l dot activation^l-1
     *
     * @param outputs list of forward propagation outputs of all the layers layer
     */
    private void updateWeights(ArrayList<LayerOutput> outputs) {
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
        return y.sub(yHat).pow(2).mul(1.0/(m*2.0));
    }


    /**
     * Derivative of the cost function.
     *
     * @param y labels
     * @param yHat predictions
     * @return error' vector
     */
    private NDArray costDerivative(NDArray y, NDArray yHat) {
        return y.sub(yHat).mul(2.0).mul(1.0/m);
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

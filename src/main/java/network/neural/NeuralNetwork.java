package network.neural;

import me.tongfei.progressbar.ProgressBar;
import network.neural.activationfunctions.IActivationFunction;
import network.neural.charts.LineChart;
import network.neural.layer.Layer;
import network.neural.layer.LayerOutput;
import network.neural.util.DataSet;
import network.neural.util.matrix.NDArray;
import network.neural.util.readwrite.ObjectIO;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;

/**
 * A multi-layer, fully connected, artificial neural network.
 * support for arbitrary number of layers with an arbitrary number of neurons.
 */
public class NeuralNetwork implements Serializable {

    private final ArrayList<Layer> layers;

    private double learningRate;
    private boolean costGraph;

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

        double[] costs; // list of costs for each epoch
        double totalTime; // total time of training

        try (ProgressBar pb = new ProgressBar("Training", epochs)) {
            long startTime = System.currentTimeMillis();
            costs = new double[epochs];


            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int batch = 0; batch < batchList.length; batch++)
                    costs[epoch] += backpropagation(batch);


                pb.step();
                pb.setExtraMessage("Cost: " + new DecimalFormat("#.##").format(costs[epoch]));
            }
            totalTime = (System.currentTimeMillis() - startTime) / 1000.0;
            System.out.println("\nTraining complete in " + totalTime + " seconds.");
        }

        return costs;
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
     * performs a single run of backward propagation on a batch
     *
     * @return the cost
     */
    private double backpropagation(int batchIndex) {

        ArrayList<LayerOutput> outputs; // the outputs of each layer.
        DataSet batch;                  // the batch of data.
        NDArray[] errors;               // the errors of each layer.
        NDArray prediction;             // the output/prediction of the network.
        double cost;                    // the cost of the batch.

        batch = batchList[batchIndex];
        outputs = forward(batch.getX());
        errors = getErrors(outputs, batchIndex);

        // assign the computed errors to each layer.
        for (int i = 0; i < layers.size(); i++)
            layers.get(i).setError(errors[i]);

        updateWeights(outputs);

        prediction = outputs.get(outputs.size() - 1).getA();
        cost = cost(batch.getY(), prediction).sum();

        return cost;
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
        NDArray[] errors;          // the errors of each layer.
        LayerOutput prediction;    // the output/prediction of the network.

        errors = new NDArray[layers.size()];
        prediction = outputs.get(outputs.size() - 1);
        errors[layers.size()-1] = getFinalLayerError(prediction, batchIndex);

        Layer currentLayer;
        Layer nextLayer;
        NDArray error;
        NDArray weights;            // the weights of the next layer.
        IActivationFunction actFun; // the activation function of the current layer.
        NDArray z;                  // output of the next layer

        for  (int i = layers.size()-2; i >= 0; i--) {
            currentLayer = layers.get(i);
            nextLayer = layers.get(i + 1);
            weights = nextLayer.getWeights();

            z = outputs.get(i+1).getZ();
            actFun = currentLayer.getActivationFunction();

            error = weights.T()
                    .dot(errors[i+1])
                    .mul(z.gradient(actFun).T());

            errors[i] = error;
        }

        return errors;
    }

    /**
     * Computes the error of the final layer.
     * error of a FinalLayer L = cost' * activation'(z^L)
     *
     * @param lastOutput the output of the last layer
     * @param batchIndex the index of the batch
     * @return the error of the final layer
     */
    private NDArray getFinalLayerError(LayerOutput lastOutput, int batchIndex) {

        Layer finalLayer;                       // the final layer of our network.
        NDArray costDerivative;                 // derivative of the cost function at the current state of the network.
        IActivationFunction activationFunction; // activation function of the last layer (Sigmoid, Relu, etc).
        NDArray z;                              // output of the last layer.


        finalLayer = layers.get(layers.size() - 1);
        costDerivative = costDerivative(batchList[batchIndex].getY(), lastOutput.getA());
        activationFunction = finalLayer.getActivationFunction();
        z = lastOutput.getZ();

        return costDerivative.mul(z.gradient(activationFunction)).T();
    }


    /**
     * updates the weights and bias of each layer
     * using the error of the layer computed in updateLayerError().
     * weights^l += error^l dot activation^l-1.
     * bias^l += error^l.
     *
     * @param outputs list of forward propagation outputs of all the layers layer
     */
    private void updateWeights(ArrayList<LayerOutput> outputs) {

        Layer layer;            // the current layer.
        NDArray weight;         // weight of the layer.
        NDArray bias;           // bias of the layer.

        NDArray error;          // error of the layer.
        NDArray a;              // activation of the layer.

        NDArray updatedWeight;  // the new weight.
        NDArray updatedBias;    // the new bias.

        for (int i = 0; i < layers.size(); i++) {
            layer = layers.get(i);
            weight = layer.getWeights();
            bias = layer.getBias();

            error = layer.getError();
            a = outputs.get(i).getA();

            updatedWeight = weight.add(error.dot(a).mul(learningRate));
            updatedBias = bias.add(error.mul(learningRate).getAvgColVector());

            layer.setWeights(updatedWeight);
            layer.setBias(updatedBias);
        }
    }


    /**
     * does forward propagation on each layer of the network
     * and returns the output of each layer
     *
     * @param x data we want to make a prediction on
     * @return the prediction
     */
    private ArrayList<LayerOutput> forward(NDArray x) {

        ArrayList<LayerOutput> outputs; // collection of all the outputs of each layer.
        LayerOutput input;             // the output of the input layer (l = 0). this layer does not contain weights.
        LayerOutput lastOutput;        // the output of the last computed layer.
        LayerOutput nextOutput;         // the output of the next layer.

        outputs = new ArrayList<>();
        input = new LayerOutput(x, x); // a and z are both the same for the input layer.
        outputs.add(input);

        for (Layer layer : layers) {
            lastOutput = outputs.get(outputs.size() - 1);
            nextOutput = layer.forward(lastOutput.getA());
            outputs.add(nextOutput);
        }

        return outputs;
    }


    /**
     * Cost function. (MSE)
     *
     * @param y labels
     * @param yHat predictions
     * @return error squared vector
     */
    private NDArray cost(NDArray y, NDArray yHat) {
        NDArray error; // error between the labels and the predictions.
        NDArray cost;  // the average cost of a training instance.
        double m;               // the number of training instances.

        m = this.m;
        error = y.sub(yHat);
        cost = error.pow(2).mul(1.0/(m*2.0));

        return cost;
    }


    /**
     * Derivative of the cost function.
     *
     * @param y labels
     * @param yHat predictions
     * @return error' vector
     */
    private NDArray costDerivative(NDArray y, NDArray yHat) {
        NDArray error;          // the error of the predictions.
        NDArray costDerivative; // the derivative of the cost function.
        double m;               // the number of training instances.

        m = this.m;
        error = y.sub(yHat);
        costDerivative = error.mul(1.0/m);

        return costDerivative;
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

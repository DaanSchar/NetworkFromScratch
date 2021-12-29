package network.neural;

import network.neural.activationfunctions.LeakyReLU;
import network.neural.activationfunctions.ReLU;
import network.neural.layer.Layer;
import network.neural.util.DataSet;
import network.neural.util.matrix.NDArray;
import network.neural.util.readwrite.ReadIO;



public class Main {

    public static void main(String[] args) {
        DataSet dataSet = DataSet.split(ReadIO.readCsv("BostonHousing.csv"), 1);

        NDArray x = dataSet.getX();
        NDArray y = dataSet.getY();

        NeuralNetwork nn = new NeuralNetwork()
                .layer(new Layer(x.shape()[1], 100, new LeakyReLU()))
                .layer(new Layer(100, 10, new LeakyReLU()))
                .layer(new Layer(10, 10, new LeakyReLU()))
                .layer(new Layer(10, y.shape()[1], new LeakyReLU()))
                .learningRate(0.00001)
                .plotCostGraph(true);

        nn.train(dataSet, 1000, 50);

    }

}

package network.neural;

import network.neural.activationfunctions.ReLU;
import network.neural.activationfunctions.Sigmoid;
import network.neural.layer.Layer;
import network.neural.util.NDArray;

public class Main {

    public static void main(String[] args) {
        NDArray input = new NDArray(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        });
        NDArray output = new NDArray(new double[][]{
                {0},
                {0},
                {0},
                {1},
        });

        NeuralNetwork nn = new NeuralNetwork(2)
                .layer(new Layer(2, 4, new ReLU()))
                .layer(new Layer(4, 10, new ReLU()))
                .layer(new Layer(10, 5, new ReLU()))
                .layer(new Layer(5, 1, new Sigmoid()))
                .learningRate(0.01)
                .plotCostGraph(true);

        nn.train(input, output, 5000);

        System.out.println(nn.predict(input));
    }
}

package network.neural;

import network.neural.activationfunctions.Linear;
import network.neural.activationfunctions.Sigmoid;

public class Main {


    public static void main(String[] args) {
        Network network = new Network(2, 3,2, new Sigmoid());

        NDArray input = new NDArray(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
        });
        NDArray output = new NDArray(new double[][]{
                {1, 0},
                {0, 1},
                {0, 1},
                {0, 1},
        });
//        output = new NDArray(new double[][]{
//                {0},
//                {1},
//                {1},
//                {0},
//        });



        network.train(input, output);
        System.out.println(network.predict(input));
//        System.out.println(network.getWeights());

    }
}

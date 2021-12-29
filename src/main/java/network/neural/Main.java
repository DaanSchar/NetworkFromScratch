package network.neural;

import network.neural.activationfunctions.LeakyReLU;
import network.neural.activationfunctions.ReLU;
import network.neural.activationfunctions.Sigmoid;
import network.neural.layer.Layer;
import network.neural.util.DataSet;
import network.neural.util.matrix.NDArray;
import network.neural.util.readwrite.ReadIO;
import org.jfree.chart.util.ArrayUtils;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;


public class Main {

    private static double x = 0;

    public static void main(String[] args) {
        DataSet dataSet = DataSet.split(ReadIO.readCsv("BostonHousing.csv"), 1);

//        dataSet = new DataSet(new NDArray(new double[][]{
//                {0, 0},
//                {0, 1},
//                {1, 0},
//                {1, 1},
//                {0, 0},
//                {0, 1},
//                {1, 0},
//                {1, 1}
//        })
//        , new NDArray(new double[][]{
//                {0},
//                {1},
//                {1},
//                {0},
//                {0},
//                {1},
//                {1},
//                {0}
//        }));

        NDArray x = dataSet.getX();
        NDArray y = dataSet.getY();


        NeuralNetwork nn = new NeuralNetwork()
                .layer(new Layer(x.shape()[1], 30, new ReLU()))
                .layer(new Layer(30, 10, new ReLU()))
                .layer(new Layer(10, 10, new ReLU()))
                .layer(new Layer(10, y.shape()[1], new ReLU()))
                .learningRate(0.00001)
                .plotCostGraph(true);

        nn.train(dataSet, 200, 5);
//        System.out.println(nn.predict(x));



    }

    static class Worker extends Thread {
        private int batch;

        public Worker(int batch) {
            this.batch = batch;
        }

        public void run() {
            try {
                sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            x+= batch;
            System.out.println("done");
        }
    }

}

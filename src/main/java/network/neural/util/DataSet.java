package network.neural.util;

import network.neural.util.matrix.NDArray;

public class DataSet {

    private NDArray x;
    private NDArray y;

    public DataSet(NDArray x, NDArray y) {
        this.x = x;
        this.y = y;
    }

    public NDArray getX() {
        return x;
    }

    public NDArray getY() {
        return y;
    }

    public void setX(NDArray x) {
        this.x = x;
    }

    public void setY(NDArray y) {
        this.y = y;
    }


    /**
     * separates the data into a feature set and a label set
     */
    public static DataSet split(NDArray data, int labelSize) {
        data = data.removeRow(0); // remove column names

        NDArray labels = data.getColumn(data.shape()[1] - 1);
        data = data.removeColumn(data.shape()[1] - 1);

        for (int i = 1; i < labelSize; i++) {
            labels = data.getColumn(data.shape()[1] - 1).concat(labels, 1);
            data = (data.removeColumn(data.shape()[1]-1));
        }

        return new DataSet(data, labels);
    }


    /**
     * separate the data into mini-batches
     */
    public static DataSet[] batch(DataSet dataSet, int batchSize) {
        int totalBatches = dataSet.getX().shape()[0] / batchSize;
        DataSet[] batches = new DataSet[totalBatches];

        for (int i = 0; i < totalBatches; i++) {
            NDArray xBatch = dataSet.getX().getRow(i * batchSize);
            NDArray yBatch = dataSet.getY().getRow(i * batchSize);

            batches[i] = new DataSet(xBatch, yBatch);
            for (int j = 1; j < batchSize; j++) {
                batches[i].setX(batches[i].getX().concat(dataSet.getX().getRow(i * batchSize + j), 0));
                batches[i].setY(batches[i].getY().concat(dataSet.getY().getRow(i * batchSize + j), 0));
            }
        }

        return batches;
    }
}

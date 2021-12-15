package network.neural;

import network.neural.activationfunctions.IActivationFunction;

import java.util.Arrays;

public class NDArray {

    private int[] shape;
    private double[][] data;

    public NDArray(double[][] data) {
        this.data = data;
        this.shape = new int[] {data.length, data[0].length};
    }

    public NDArray(double[] data) {
        this.data = new double[1][data.length];
        this.data[0] = data;
        this.shape = new int[] {1, data.length};
    }

    /**
     * Transposes a NDArray
     * @return A new NDArray with the transposed data
     */
    public NDArray T() {
        double[][] newData = new double[data[0].length][data.length];

        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                newData[j][i] = data[i][j];

        return new NDArray(newData);
    }

    /**
     * performs matrix dot product on two NDArrays
     *
     * @param other The NDArray to multiply this instance by
     * @return A new NDArray with the result of the matrix dot product
     */
    public NDArray dot(NDArray other) {
        if (this.shape[1] != other.shape[0])
            throw new IllegalArgumentException("Incompatible shapes");

        int cols = other.shape[1];
        int rows = this.shape[0];

        double[][] newData = new double[rows][cols];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j <cols; j++)
                for (int k = 0; k < this.shape[1]; k++)
                    newData[i][j] += this.data[i][k] * other.data[k][j];

        return new NDArray(newData);
    }

    /**
     * multiplies each index of the NDArray by a scalar
     *
     * @param scalar The scalar to multiply each index by
     * @return A new NDArray with each index multiplied by the scalar
     */
    public NDArray mul(double scalar) {
        double[][] result = new double[shape[0]][shape[1]];
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                result[i][j] = data[i][j] * scalar;

        return new NDArray(result);
    }

    public NDArray pow(int power) {
        double[][] newData = new double[shape[0]][shape[1]];

        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                newData[i][j] = Math.pow(data[i][j], power);

        return new NDArray(newData);
    }

    /**
     * Sum of all elements in the NDArray
     * @return The sum of all the elements in the NDArray
     */
    public double sum() {
        double sum = 0;
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                sum += data[i][j];

        return sum;
    }

    /**
     * Adds two NDArrays together
     * @param other The NDArray to add to this instance
     * @return A new NDArray with the result of the addition
     */
    public NDArray add(NDArray other) {
        if (this.shape[0] != other.shape[0] || this.shape[1] != other.shape[1])
            throw new IllegalArgumentException("Incompatible shapes");

        double[][] newData = new double[this.shape[0]][this.shape[1]];

        for (int i = 0; i < this.shape[0]; i++)
            for (int j = 0; j < this.shape[1]; j++)
                newData[i][j] = this.data[i][j] + other.data[i][j];

        return new NDArray(newData);
    }

    /**
     * Subtracts two NDArrays
     * @param other The NDArray to subtract from this instance
     * @return A new NDArray with the result of the subtraction
     */
    public NDArray sub(NDArray other) {
        return this.add(other.mul(-1));
    }


    public double get(int i, int j) {
        return data[i][j];
    }

    public double[][] data() {
        return data;
    }

    public int[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }


    /**
     * runs an NDArray through a given activation function
     *
     * @param function The activation function to apply
     * @return A new NDArray with the given activation function applied to each index
     */
    public NDArray activation(IActivationFunction function) {
        double[][] result = new double[shape[0]][shape[1]];
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                result[i][j] = function.get(data[i][j]);

        return new NDArray(result);
    }

    /**
     * calculates the derivative of the activation function for each index
     *
     * @param function The activation function to apply
     * @return A new NDArray with the derivative of the activation function applied to each index
     */
    public NDArray gradient(IActivationFunction function) {
        double[][] result = new double[shape[0]][shape[1]];
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++)
                result[i][j] = function.gradient(data[i][j]);

        return new NDArray(result);
    }


    @Override
    public String toString() {
        String data = "[";
        for (int i = 0; i < this.data.length; i++) {
            String nextLine = i  == this.data.length - 1 ? "" : "\n ";
            data += Arrays.toString(this.data[i]) + nextLine;
        }

        return "shape=" + Arrays.toString(shape) + "\n" + data + "]";
    }


    /**
     * NDArray filled with zeros
     *
     * @param shape The shape of the NDArray
     * @return A new NDArray filled with zeros
     */
    public static NDArray zeros(int... shape) {
        if (shape.length > 2 || shape.length < 1)
            throw new IllegalArgumentException("Invalid shape, must be 1 or 2 dimensions");

        double[][] data = new double[shape[0]][shape[1]];
        return new NDArray(data);
    }


    /**
     * NDArray filled with ones
     *
     * @param shape The shape of the NDArray
     * @return A new NDArray filled with ones
     */
    public static NDArray ones(int... shape) {
        if (shape.length > 2 || shape.length < 1)
            throw new IllegalArgumentException("Invalid shape, must be 1 or 2 dimensions");

        double[][] data = new double[shape[0]][shape[1]];

        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                data[i][j] = 1;

        return new NDArray(data);
    }

    /**
     * random initialisation of the NDArray
     *
     * @param shape The shape of the NDArray
     * @return A new NDArray with random values
     */
    public static NDArray rand(int... shape) {
        if (shape.length > 2 || shape.length < 1)
            throw new IllegalArgumentException("Invalid shape, must be 1 or 2 dimensions");

        double[][] data = new double[shape[0]][shape[1]];

        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                data[i][j] = Math.random();

        return new NDArray(data);
    }

}

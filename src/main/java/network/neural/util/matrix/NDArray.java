package network.neural.util.matrix;

import network.neural.activationfunctions.IActivationFunction;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

public class NDArray implements Serializable {

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


    public NDArray getRow(int row) {
        return new NDArray(new double[][] {data[row]});
    }


    public NDArray getColumn(int col) {
        double[][] newData = new double[shape[0]][1];

        for (int i = 0; i < shape[0]; i++)
            newData[i][0] = data[i][col];

        return new NDArray(newData);
    }


    /**
     * removes a row from the NDArray
     * @param i The index of the row to remove
     * @return A new NDArray with the row removed
     */
    public NDArray removeRow(int i) {
        double[][] newData = new double[shape[0] - 1][shape[1]];

        for (int j = 0; j < shape[0] - 1; j++)
            if (j < i)
                newData[j] = data[j];
            else
                newData[j] = data[j+1];

        return new NDArray(newData);
    }


    /**
     * removes a column from the NDArray
     * @param i The index of the column to remove
     * @return A new NDArray with the column removed
     */
    public NDArray removeColumn(int i) {
        double[][] newData = new double[shape[0]][shape[1] - 1];
        double[] newRow;

        for (int k = 0; k < shape[0]; k++) {
             newRow = new double[shape[1] - 1];

            for (int j = 0; j < shape[1] - 1; j++) {
                if (j < i)
                    newRow[j] = data[k][j];
                else
                    newRow[j] = data[k][j + 1];
            }
            newData[k] = newRow;
        }
        return new NDArray(newData);
    }


    public NDArray concat(NDArray other, int axis) {
        if (axis > 1)
            throw new IllegalArgumentException("Axis must be 0 or 1");

        if (this.shape[1] != other.shape[1] && axis == 0)
            throw new IllegalArgumentException("Incompatible shapes" + Arrays.toString(this.shape) + " and " + Arrays.toString(other.shape));

        if (this.shape[0] != other.shape[0] && axis == 1)
            throw new IllegalArgumentException("Incompatible shapes" + Arrays.toString(this.shape) + " and " + Arrays.toString(other.shape));


        NDArray newData = null;

        if (axis == 0)
             newData = concatRows(other);
        if (axis == 1)
            newData = concatCols(other);

        return newData;
    }


    /**
     * Concatenates two NDArrays along the rows
     */
    private NDArray concatRows(NDArray other) {
        double[][] newData = new double[this.shape[0] + other.shape[0]][this.shape[1]];

        for (int i = 0; i < this.shape[0]; i++)
            newData[i] = this.data[i];

        for (int i = 0; i < other.shape[0]; i++)
            newData[i + this.shape[0]] = other.data[i];

        return new NDArray(newData);
    }


    /**
     * Adds two NDArrays along the columns
     */
    private NDArray concatCols(NDArray other) {
        double[][] newData = new double[this.shape[0]][this.shape[1] + other.shape[1]];

        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++)
                newData[i][j] = this.data[i][j];
            for (int j = 0; j < other.shape[1]; j++)
                newData[i][j + this.shape[1]] = other.data[i][j];
        }

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
            throw new IllegalArgumentException("Incompatible shapes" + Arrays.toString(this.shape) + " and " + Arrays.toString(other.shape));

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


    public NDArray mul(NDArray other) {
        if (this.shape[0] != other.shape[0] || this.shape[1] != other.shape[1])
            throw new IllegalArgumentException("Incompatible shapes " + Arrays.toString(this.shape) + " and " + Arrays.toString(other.shape));

        double[][] newData = new double[this.shape[0]][this.shape[1]];

        for (int i = 0; i < this.shape[0]; i++)
            for (int j = 0; j < this.shape[1]; j++)
                newData[i][j] = this.data[i][j] * other.data[i][j];

        return new NDArray(newData);
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
            throw new IllegalArgumentException("Incompatible shapes " + Arrays.toString(this.shape) + " and " + Arrays.toString(other.shape()));

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


    public NDArray addVector(NDArray vector) {
        if (vector.shape[0] != 1 && vector.shape[1] != 1)
            throw new IllegalArgumentException("Incompatible shapes. " + Arrays.toString(vector.shape) + " is not a vector");
        if (vector.shape[0] == 1 && this.shape[1] == vector.shape[1])
            return this.addRowVector(vector);
        if (vector.shape[1] == 1)
            return this.addColumnVector(vector);

        return null;
    }


    private NDArray addRowVector(NDArray vector) {
        if (this.shape[1] != vector.shape[1])
            throw new IllegalArgumentException("Incompatible shapes " + Arrays.toString(this.shape) + " and " + Arrays.toString(vector.shape()));

        double[][] result = new double[this.shape[0]][ this.shape[1]];

        for (int i = 0; i < this.shape[0]; i++)
            for (int j = 0; j < this.shape[1]; j++)
                result[i][j] = this.data[i][j] + vector.data[0][j];

        return new NDArray(result);
    }


    private NDArray addColumnVector(NDArray vector) {
        if (this.shape[0] != vector.shape[0])
            throw new IllegalArgumentException("Incompatible shapes " + Arrays.toString(this.shape) + " and " + Arrays.toString(vector.shape()));

        double[][] result = new double[this.shape[0]][ this.shape[1]];

        for (int i = 0; i < this.shape[0]; i++)
            for (int j = 0; j < this.shape[1]; j++)
                result[i][j] = this.data[i][j] + vector.data[i][0];

        return new NDArray(result);
    }


    public NDArray getAvgColVector() {
        double[][] result = new double[this.shape[0]][1];

        for (int i = 0; i < this.shape[0]; i++)
            result[i][0] = this.getRow(i).sum() / this.shape[1];

        return new NDArray(result);
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


    @Override
    public String toString() {
        StringBuilder data = new StringBuilder("[");
        for (int i = 0; i < this.data.length; i++) {
            String nextLine = i  == this.data.length - 1 ? "" : "\n ";
            data.append(Arrays.toString(this.data[i])).append(nextLine);
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


    /**
     * random initialisation of the NDArray
     *
     * @param shape The shape of the NDArray
     * @return A new NDArray with random values between the values
     *         max and -max
     */
    public static NDArray rand(double max, int... shape) {
        if (shape.length > 2 || shape.length < 1)
            throw new IllegalArgumentException("Invalid shape, must be 1 or 2 dimensions");

        double[][] data = new double[shape[0]][shape[1]];

        for (int i = 0; i < data.length; i++)
            for (int j = 0; j < data[0].length; j++)
                data[i][j] = Math.random() * max - max/2.0;

        return new NDArray(data);
    }

}

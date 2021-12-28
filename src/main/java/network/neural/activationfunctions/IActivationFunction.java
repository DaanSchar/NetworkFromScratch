package network.neural.activationfunctions;

public interface IActivationFunction{

    /**
     * Applies the activation function to the given input.
     * @param x double value to apply the activation function to
     * @return output of the activation function
     */
    double get(double x);


    /**
     * calculates the gradient of the activation function given an input.
     * @param x double to apply the activation function to
     * @return output of the activation function
     */
    double gradient(double x);
}


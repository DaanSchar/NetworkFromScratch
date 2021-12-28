package network.neural.activationfunctions;

import java.io.Serializable;

public class Tanh implements IActivationFunction, Serializable {
    @Override
    public double get(double x) {
        return Math.tanh(x);
    }

    @Override
    public double gradient(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }


}

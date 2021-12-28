package network.neural.activationfunctions;

import java.io.Serializable;

public class ReLU implements IActivationFunction, Serializable {

    @Override
    public double get(double x) {
        return Math.max(0, x);
    }

    @Override
    public double gradient(double x) {
        return x > 0 ? 1 : 0;
    }

}

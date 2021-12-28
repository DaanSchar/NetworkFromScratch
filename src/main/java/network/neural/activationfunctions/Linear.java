package network.neural.activationfunctions;

import java.io.Serializable;

public class Linear implements IActivationFunction, Serializable {
    @Override
    public double get(double x) {
        return x;
    }

    @Override
    public double gradient(double x) {
        return 1;
    }

}

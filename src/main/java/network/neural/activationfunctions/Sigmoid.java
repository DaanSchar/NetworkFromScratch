package network.neural.activationfunctions;

import network.neural.NDArray;

public class Sigmoid implements IActivationFunction {
    @Override
    public double get(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double gradient(double x) {
        return get(x) * (1.0 - get(x));
    }

}

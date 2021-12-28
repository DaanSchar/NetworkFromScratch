package network.neural.activationfunctions;

public class LeakyReLU implements IActivationFunction{
    @Override
    public double get(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    @Override
    public double gradient(double x) {
        return x > 0 ? 1 : 0.01;
    }
}

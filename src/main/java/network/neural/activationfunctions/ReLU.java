package network.neural.activationfunctions;

public class ReLU implements IActivationFunction{

    @Override
    public double get(double x) {
        return Math.max(0, x);
    }

    @Override
    public double gradient(double x) {
        return x > 0 ? 1 : 0;
    }

}

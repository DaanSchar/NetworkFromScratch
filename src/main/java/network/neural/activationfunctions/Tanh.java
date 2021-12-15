package network.neural.activationfunctions;

public class Tanh implements IActivationFunction{
    @Override
    public double get(double x) {
        return Math.tanh(x);
    }

    @Override
    public double gradient(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }


}

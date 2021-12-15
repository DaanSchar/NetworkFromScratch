package network.neural.activationfunctions;

public class Linear implements IActivationFunction{
    @Override
    public double get(double x) {
        return x;
    }

    @Override
    public double gradient(double x) {
        return 1;
    }

}

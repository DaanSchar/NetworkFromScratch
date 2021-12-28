package network.neural;

public class LayerOutput {

    private NDArray z;
    private NDArray a;

    public LayerOutput(NDArray z, NDArray a) {
        this.z = z;
        this.a = a;
    }

    public NDArray getZ() {
        return z;
    }

    public NDArray getA() {
        return a;
    }
}

package network.neural.layer;

import network.neural.util.matrix.NDArray;

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

import network.neural.NDArray;
import network.neural.Network;
import network.neural.activationfunctions.Linear;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class FeedForwardTest {

    @Test
    public void feedforward1() {
        Network network = new Network(2, 2, 1, new Linear());

        network.setWeights(new NDArray(new double[][] {
                {5, 2},
                {3, 6}
        }));
        network.setHiddenWeights(new NDArray(new double[][] {
                {1, 0},
                {0, 1}
        }));
        network.setBiases(new NDArray(new double[][] {{6}, {6}}));
        network.setHiddenBiases(new NDArray(new double[][] {{0}, {0}}));
        NDArray result = network.predict(new NDArray(new double[][] {{10}, {20}}));
        assertEquals(96,result.data()[0][0]);
        assertEquals(156,result.data()[1][0]);

    }
}

import network.neural.util.matrix.NDArray;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NDArrayTransformTest {

    @Test
    public void testTransform1() {
        NDArray nd = new NDArray(new double[] {2, 3, 4});

        for (int i = 0; i < nd.shape()[0]; i++) {
            assertEquals(nd.T().get(i, 0), nd.get(0, i));
        }

    }

    @Test
    public void testTransform2() {
        NDArray nd = new NDArray(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

        for (int i = 0; i < nd.shape()[0]; i++)
            for (int j = 0; j < nd.shape()[1]; j++)
                assertEquals(nd.T().get(j, i), nd.get(i, j));
    }

    @Test
    public void testTransform3() {
        NDArray nd = new NDArray(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});

        for (int i = 0; i < nd.shape()[0]; i++)
            for (int j = 0; j < nd.shape()[1]; j++)
                assertEquals(nd.T().get(j, i), nd.get(i, j));
    }



}

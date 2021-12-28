import network.neural.util.matrix.NDArray;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NDArrayShapeTest {

    @Test
    public void testShape1() {
        NDArray nd = new NDArray(new double[] {2, 3, 4});
        assertEquals(1, nd.shape()[0]);
        assertEquals(3, nd.shape()[1]);
    }

    @Test
    public void testShape2() {
        NDArray nd = new NDArray(new double[] {-1, -2, -3, 4, 5, 6});
        assertEquals(1, nd.shape()[0]);
        assertEquals(6, nd.shape()[1]);
    }

    @Test
    public void testShape3() {
        NDArray nd = new NDArray(new double[][] {{1}, {2}, {3}});
        assertEquals(3, nd.shape()[0]);
        assertEquals(1, nd.shape()[1]);
    }

    @Test
    public void testShape4() {
        NDArray nd = new NDArray(new double[][] {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}});
        assertEquals(3, nd.shape()[0]);
        assertEquals(3, nd.shape()[1]);
    }

    @Test
    public void testShape5() {
        NDArray nd = new NDArray(new double[][] {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}, {5, 6, 7}});
        assertEquals(5, nd.shape()[0]);
        assertEquals(3, nd.shape()[1]);
    }

}

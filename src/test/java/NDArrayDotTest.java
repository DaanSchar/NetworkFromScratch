import network.neural.util.NDArray;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NDArrayDotTest {

    @Test
    public void testDot1() {
        NDArray nd = new NDArray(new double[][] {{1},  {2}, {3}});
        NDArray nd2 = new NDArray(new double[] {1, 2, 3});
        NDArray nd3 = nd.dot(nd2);
        assertEquals(1, nd3.get(0, 0));
        assertEquals(2, nd3.get(0, 1));
        assertEquals(3, nd3.get(0, 2));
        assertEquals(2, nd3.get(1, 0));
        assertEquals(4, nd3.get(1, 1));
        assertEquals(6, nd3.get(1, 2));
        assertEquals(3, nd3.get(2, 0));
        assertEquals(6, nd3.get(2, 1));
        assertEquals(9, nd3.get(2, 2));
    }

    @Test
    public void testDot2() {
        NDArray nd = new NDArray(new double[] {1, 2, 3});
        NDArray nd2 = new NDArray(new double[][] {{1}, {2}, {3}});
        NDArray nd3 = nd.dot(nd2);
        assertEquals(14, nd3.get(0, 0));
    }

    @Test
    public void testDot3() {
        NDArray nd = new NDArray(new double[][] {{1, 2, 3}, {4, 5, 6}});
        NDArray nd2 = new NDArray(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        NDArray nd3 = nd.dot(nd2);
        assertEquals(22, nd3.get(0, 0));
        assertEquals(28, nd3.get(0, 1));
        assertEquals(49, nd3.get(1, 0));
        assertEquals(64, nd3.get(1, 1));
    }

    @Test
    public void testDot4() {
        NDArray nd = new NDArray(new double[][] {{1, 2, 3}, {4, 5, 6}});
        NDArray nd2 = new NDArray(new double[][] {{7, 8}, {9, 10}, {11, 12}});
        NDArray result = nd.dot(nd2);
        assertEquals(58, result.get(0, 0));
        assertEquals(64, result.get(0, 1));
        assertEquals(139, result.get(1, 0));
        assertEquals(154, result.get(1, 1));
    }

    @Test
    public void testDot5() {
        NDArray nd = new NDArray(new double[][] {{1, 2, 3}});
        NDArray nd2 = new NDArray(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        NDArray nd3 = nd.dot(nd2);
        assertEquals(22, nd3.get(0, 0));
        assertEquals(28, nd3.get(0, 1));
    }

    @Test
    public void testDot6() {
        NDArray nd = new NDArray(new double[][] {{1, 2, 3}, {4, 5, 6}});
        NDArray nd2 = new NDArray(new double[][]{{1}, {2}, {3}});
        NDArray nd3 = nd.dot(nd2);
        assertEquals(14, nd3.get(0, 0));
        assertEquals(32, nd3.get(1, 0));
    }

}

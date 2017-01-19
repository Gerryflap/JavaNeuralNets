package nl.gerben_meijer.neuralnets.math.functions;

/**
 * Created by gerben on 17-1-17.
 */
public class ReLU implements Function {
    @Override
    public float apply(float x) {
        return Math.max(0, x);
    }
}

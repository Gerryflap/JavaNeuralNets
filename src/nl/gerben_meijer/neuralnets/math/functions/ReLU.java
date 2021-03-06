package nl.gerben_meijer.neuralnets.math.functions;

/**
 * Created by gerben on 17-1-17.
 */
public class ReLU implements Function {
    @Override
    public float apply(float x) {
        if (x > 0) {
            return x;
        } else {
            return 0.01f * x;
        }
    }
}

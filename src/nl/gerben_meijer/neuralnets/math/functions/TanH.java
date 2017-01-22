package nl.gerben_meijer.neuralnets.math.functions;

/**
 * Created by gerben on 17-1-17.
 */
public class TanH implements Function{


    @Override
    public float apply(float x) {
        return (float) Math.tanh(x);
    }
}

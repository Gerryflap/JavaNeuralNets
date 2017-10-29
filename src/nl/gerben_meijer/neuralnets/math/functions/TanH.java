package nl.gerben_meijer.neuralnets.math.functions;

/**
 * Created by gerben on 17-1-17.
 */
public class TanH implements GradientFunction{


    @Override
    public float apply(float x) {
        return (float) Math.tanh(x);
    }

    @Override
    public float gradient(float x) {

        return 1 - apply(x) * apply(x);
    }
}

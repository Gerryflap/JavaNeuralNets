package nl.gerben_meijer.neuralnets.math.functions;

/**
 * Created by gerben on 23-12-16.
 * The sigmoid function
 */
public class Sigmoid implements GradientFunction {
    @Override
    public float apply(float x) {
        return (float) (1/(1+Math.exp(x)));
    }

    @Override
    public float gradient(float x) {
        return apply(x) * (1 - apply(x));
    }
}

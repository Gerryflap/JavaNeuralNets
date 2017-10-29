package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.Matrix;

import java.util.Collection;
import java.util.LinkedList;

/**
 * Created by gerben on 27-7-17.
 * TODO: Implement
 */
public class MaxPool extends Layer {

    @Override
    public Matrix forwardPass(Matrix input) {
        return null;
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        return new LinkedList<>();
    }
}

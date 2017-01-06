package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.Matrix;

import java.io.Serializable;
import java.util.Collection;

/**
 * Created by gerben on 23-12-16.
 * Models a neural layer
 */
public interface Layer extends Serializable {

    Matrix forwardPass(Matrix input);

    /**
     * Collects all variables that can be changed by the optimizer
     * @return Collection of matrices that can be changed.
     */
    Collection<Matrix> getFreeVariables();
}

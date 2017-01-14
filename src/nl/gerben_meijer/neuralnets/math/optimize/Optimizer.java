package nl.gerben_meijer.neuralnets.math.optimize;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;

/**
 * Created by gerben on 14-1-17.
 *
 * Models an Optimizer
 */
public interface Optimizer {

    public void optimize(Matrix inputBatch, Matrix correctBatch) throws InvalidDimensionsException;
}

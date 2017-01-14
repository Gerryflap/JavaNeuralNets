package nl.gerben_meijer.neuralnets.math.functions;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;

/**
 * Created by gerben on 23-12-16.
 * Models a cost function.
 */
public interface CostFunction {

    double apply(Matrix output, Matrix correct) throws InvalidDimensionsException, InvalidDimensionsException;

}

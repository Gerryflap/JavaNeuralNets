package nl.gerben_meijer.neuralnets.math.functions;

import nl.gerben_meijer.neuralnets.math.Matrix;

/**
 * Created by gerben on 23-12-16.
 * Models a cost function.
 */
public interface CostFunction {

    double apply(Matrix output, Matrix correct) throws Matrix.InvalidDimensionsException;

}

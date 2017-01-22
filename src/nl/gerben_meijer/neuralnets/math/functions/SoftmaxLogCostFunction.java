package nl.gerben_meijer.neuralnets.math.functions;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;

/**
 * Created by gerben on 23-12-16.
 */
public class SoftmaxLogCostFunction implements CostFunction {


    @Override
    public double apply(Matrix output, Matrix correct) throws InvalidDimensionsException {

        float[] ratings = new float[output.getWidth()];

        // Create a squared delta matrix
        Matrix diff = correct.mapFunction(x->1-x).add(output.mapFunction(x -> -x)).mapFunction(x -> (float) -Math.log(Math.abs(x)));
        return diff.sum();
    }
}

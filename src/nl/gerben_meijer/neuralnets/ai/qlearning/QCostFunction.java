package nl.gerben_meijer.neuralnets.ai.qlearning;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;

/**
 * Created by gerben on 13-1-17.
 */
public class QCostFunction implements CostFunction {

    @Override
    public double apply(Matrix output, Matrix correct) throws InvalidDimensionsException {
        double cost = output.add(correct.mapFunction(x->-x)).mapFunction(x-> x*x).sum();
        return cost;
    }

}

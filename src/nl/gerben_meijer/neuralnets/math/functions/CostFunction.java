package nl.gerben_meijer.neuralnets.math.functions;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.NetworkInput;
import nl.gerben_meijer.neuralnets.math.Sequence;

import java.util.Iterator;

/**
 * Created by gerben on 23-12-16.
 * Models a cost function.
 */
public abstract class CostFunction {

    public abstract double apply(Matrix output, Matrix correct) throws InvalidDimensionsException;

    public double apply(Sequence output, Sequence correct) throws InvalidDimensionsException {
        Iterator<Matrix> iterator = output.getIterator();
        Iterator<Matrix> correctIterator = correct.getIterator();
        double totalCost = 0;
        while (iterator.hasNext()) {
            Matrix cOutp = iterator.next();
            Matrix cCorr = correctIterator.next();
            totalCost += apply(cOutp, cCorr);
        }
        return totalCost/output.size();
    }

    public double apply(NetworkInput output, NetworkInput correct) throws InvalidDimensionsException {
        if (output instanceof Matrix) {
            return apply((Matrix) output, (Matrix) correct);
        } else {
            return apply((Sequence) output, (Sequence) correct);
        }
    }

}

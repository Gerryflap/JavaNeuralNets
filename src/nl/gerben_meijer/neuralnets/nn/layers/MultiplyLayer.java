package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Collection;
import java.util.LinkedList;

/**
 * Created by gerben on 23-12-16.
 */
public class MultiplyLayer extends Layer{
    private Matrix mult = new Matrix(1,1);


    @Override
    public Matrix forwardPass(Matrix input) {
        return input.mapFunction(x -> mult.getValue(0,0) * x);
    }

    @Override
    public Matrix backPropagate(Matrix input, Matrix error) throws InvalidDimensionsException {
        throw new NotImplementedException();
    }

    @Override
    public void updateVars(Matrix input, Matrix error) {
        throw new NotImplementedException();
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        LinkedList<Matrix> out = new LinkedList<>();
        out.add(mult);
        return out;
    }
}

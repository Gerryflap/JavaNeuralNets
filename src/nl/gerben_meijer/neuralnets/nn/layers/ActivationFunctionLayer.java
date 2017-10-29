package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.Function;
import nl.gerben_meijer.neuralnets.math.functions.GradientFunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by gerben on 23-12-16.
 */
public class ActivationFunctionLayer extends Layer{
    private Function function;

    public ActivationFunctionLayer(Function function) {
        this.function = function;
    }

    @Override
    public Matrix forwardPass(Matrix input) {
        return input.mapFunction(function);
    }

    @Override
    public Matrix backPropagate(Matrix input, Matrix error) throws InvalidDimensionsException {
        if (function instanceof GradientFunction) {
            return input.mapFunction(x -> ((GradientFunction) function).gradient(x)).multiply(error);
        }
        throw new RuntimeException("Function does not have a gradient!");
    }

    @Override
    public void updateVars(Matrix input, Matrix error) {
        // This method has no weights
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        return new LinkedList<>();
    }
}

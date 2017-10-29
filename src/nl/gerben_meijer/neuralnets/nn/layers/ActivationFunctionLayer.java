package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.Function;

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
    public Collection<Matrix> getFreeVariables() {
        return new LinkedList<>();
    }
}

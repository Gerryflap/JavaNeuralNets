package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.Sequence;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;

/**
 * Created by gerben on 23-12-16.
 * Models a neural layer
 */
public abstract class Layer implements Serializable {

    public abstract Matrix forwardPass(Matrix input);

    public Sequence forwardPass(Sequence input) {
        Iterator<Matrix> iterator = input.getIterator();
        Sequence out = new Sequence();
        while (iterator.hasNext()) {
            Matrix im = iterator.next();
            out.addMatrix(forwardPass(im));
        }
        return out;
    }

    public abstract Matrix backPropagate(Matrix input, Matrix error) throws InvalidDimensionsException;

    public abstract void updateVars(Matrix input, Matrix error);

    /**
     * Collects all variables that can be changed by the optimizer
     * @return Collection of matrices that can be changed.
     */
    public abstract Collection<Matrix> getFreeVariables();
}

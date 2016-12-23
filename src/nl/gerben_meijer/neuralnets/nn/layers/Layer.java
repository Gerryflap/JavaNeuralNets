package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.Matrix;

/**
 * Created by gerben on 23-12-16.
 * Models a neural layer
 */
public interface Layer {

    Matrix forwardPass(Matrix input);
}

package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.Matrix;

import java.util.Collection;
import java.util.LinkedList;

/**
 * Created by gerben on 23-12-16.
 */
public class FullyConnectedLayer implements Layer {

    private Matrix weights;
    private Matrix biases;

    public FullyConnectedLayer(Matrix weights, Matrix biases) {
        this.weights = weights;
        this.biases = biases;
    }

    public FullyConnectedLayer (int inputs, int outputs) {
        weights = Matrix.initRandom(inputs, outputs);
        biases = Matrix.initRandom(1, outputs);
    }

    @Override
    public Matrix forwardPass(Matrix input) {
        Matrix out = null;
        try {
            out = input.matmul(weights).addAsVector(biases);
        } catch (Matrix.InvalidDimensionsException e) {
            e.printStackTrace();
        }
        return out;
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        Collection<Matrix> out = new LinkedList<>();
        out.add(weights);
        out.add(biases);
        return out;
    }

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getBiases() {
        return biases;
    }
}

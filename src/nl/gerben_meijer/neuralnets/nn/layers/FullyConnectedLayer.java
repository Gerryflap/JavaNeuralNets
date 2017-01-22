package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;

import java.util.Collection;
import java.util.LinkedList;

/**
 * Created by gerben on 23-12-16.
 */
public class FullyConnectedLayer implements Layer {

    public enum InitOption {GAUSSIAN, TANH_SCALED_RANDOM, SIGMOID_SCALED_GAUSSIAN, ZEROES}
    private Matrix weights;
    private Matrix biases;

    public FullyConnectedLayer(Matrix weights, Matrix biases) {
        this.weights = weights;
        this.biases = biases;
    }

    public FullyConnectedLayer (int inputs, int outputs) {
        this(inputs, outputs, InitOption.TANH_SCALED_RANDOM);
    }

    public FullyConnectedLayer(int inputs, int outputs, InitOption initOption) {
        switch (initOption) {
            case GAUSSIAN:
                weights = Matrix.initRandom(inputs, outputs);
                biases = Matrix.initRandom(1, outputs);
                break;
            case TANH_SCALED_RANDOM:
                weights = Matrix.initRandom(inputs, outputs, Math.sqrt(6f/(inputs + outputs)));
                biases = Matrix.initRandom(1, outputs, Math.sqrt(6f/(inputs + outputs)));
                break;
            case SIGMOID_SCALED_GAUSSIAN:
                weights = Matrix.initRandom(inputs, outputs, 4*Math.sqrt(6f/(inputs + outputs)));
                biases = Matrix.initRandom(1, outputs, 4*Math.sqrt(6f/(inputs + outputs)));
                break;
            case ZEROES:
                weights = new Matrix(inputs, outputs);
                biases = new Matrix(1, outputs);
                break;
        }
    }

    public FullyConnectedLayer (int inputs, int outputs, int constant) {
        if (constant == 0) {
            weights = new Matrix(inputs, outputs);
            biases = new Matrix(1, outputs);
        } else {
            weights = new Matrix(inputs, outputs).mapFunction(x->constant);
            biases = new Matrix(1, outputs).mapFunction(x->constant);
        }
    }

    @Override
    public Matrix forwardPass(Matrix input) {
        Matrix out = null;
        try {
            out = input.matmul(weights).addAsVector(biases);
        } catch (InvalidDimensionsException e) {
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

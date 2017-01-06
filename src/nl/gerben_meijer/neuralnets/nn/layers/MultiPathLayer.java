package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.Matrix;

import java.util.Collection;
import java.util.LinkedList;

/**
 * Created by gerben on 6-1-17.
 * A layer consisting of multiple separated channels, each with their own fc neural network.
 * All subnets receive the same input and have the same output size.
 * The total output matrix has a height of output size times the number of channels
 */
public class MultiPathLayer implements Layer{

    private int channels;
    private int outputSize;

    private FullyConnectedLayer[] layers;

    public MultiPathLayer(int inputSize, int outputSize, int channels) {
        this.channels = channels;
        this.outputSize = outputSize;

        layers = new FullyConnectedLayer[channels];
        for (int i = 0; i < channels; i++) {
            layers[i] = new FullyConnectedLayer(inputSize, outputSize);
        }

    }

    @Override
    public Matrix forwardPass(Matrix input) {
        float[][] out = new float[channels*outputSize][input.getWidth()];

        for (int i = 0; i < layers.length; i++) {
            float[][] outp = layers[i].forwardPass(input).getData();
            System.arraycopy(outp, 0, out, i * outputSize, outputSize);
        }


        try {
            return new Matrix(out);
        } catch (Matrix.InvalidDimensionsException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        Collection<Matrix> free = new LinkedList<>();
        for (FullyConnectedLayer layer:
             layers) {
            free.addAll(layer.getFreeVariables());
        }
        return free;
    }
}

package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.Matrix;

import java.util.Collection;
import java.util.LinkedList;

/**
 * Created by gerben on 6-1-17.
 * A layer consisting of multiple separated channels, each with their own fc neural network.
 * All subnets receive the same input and have the same output size.
 * The total output matrix has a height of output size times the number of channels
 * TODO: Implement Convolution
 */
public class ConvolutionLayer implements Layer{

    private int channels;
    private int kernelSize;
    private int inputWidth;
    private int inputHeight;
    private int inputDepth;
    private FullyConnectedLayer[] layers;

    public ConvolutionLayer(int channels, int kernelSize, int inputWidth, int inputHeight, int inputDepth) {
        this.channels = channels;
        this.kernelSize = kernelSize;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        layers = new FullyConnectedLayer[channels];
        for (int i = 0; i < channels; i++) {
            layers[i] = new FullyConnectedLayer(inputWidth * inputHeight * inputDepth, 1);
        }
    }





    @Override
    public Matrix forwardPass(Matrix input) {
        float[][] out = new float[channels*inputHeight*inputWidth][input.getWidth()];

        for (int xOffset = kernelSize; xOffset < inputWidth - kernelSize; xOffset++) {
            for (int yOffset = kernelSize; yOffset < inputHeight - kernelSize; yOffset++) {
                for (int zOffset = kernelSize; zOffset < inputDepth - kernelSize; zOffset++) {
                    // TODO: Generate kernelData and forwardpass network
                    float[][] kernelData = new float[kernelSize][kernelSize];
                }


            }

        }

        for (int i = 0; i < layers.length; i++) {
            float[][] outp = layers[i].forwardPass(input).getData();
            //System.arraycopy(outp, 0, out, i * outputSize, outputSize);
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

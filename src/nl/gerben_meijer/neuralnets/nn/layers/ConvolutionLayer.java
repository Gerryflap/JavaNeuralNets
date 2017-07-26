package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

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

    private int outputChannels;
    private int kernelSize;
    private int inputWidth;
    private int inputHeight;
    private int inputChannels;
    private FullyConnectedLayer[][] layers;

    public ConvolutionLayer(int outputChannels, int kernelSize, int inputWidth, int inputHeight, int inputChannels) {
        this.outputChannels = outputChannels;
        this.kernelSize = kernelSize;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputChannels = inputChannels;

        layers = new FullyConnectedLayer[inputChannels][outputChannels];
        for (int i = 0; i < inputChannels; i++) {
            for (int o = 0; o < outputChannels; o++) {
                layers[i][o] = new FullyConnectedLayer(kernelSize*kernelSize, 1);
            }
        }
    }





    @Override
    public Matrix forwardPass(Matrix input) {
        if (inputWidth * inputHeight * inputChannels != input.getHeight()) {
            try {
                throw new InvalidDimensionsException(String.format("Expected an input of %d, got %d", inputWidth * inputHeight * inputChannels, input.getHeight()));
            } catch (InvalidDimensionsException e) {
                e.printStackTrace();
            }
        }

        Matrix out = new Matrix(input.getWidth(), (inputWidth - kernelSize+1) * (inputHeight - kernelSize+1) * outputChannels);


        for (int xOffset = 0; xOffset < inputWidth - kernelSize; xOffset++) {
            for (int yOffset = 0; yOffset < inputHeight - kernelSize; yOffset++) {
                for (int inChan = 0; inChan < inputChannels; inChan++) {
                    for (int outChan = 0; outChan < outputChannels; outChan++) {
                        Matrix localOut = layers[inChan][outChan].forwardPass(
                                getKernelInput(input, xOffset, yOffset, inChan)
                        );
                        //System.out.println(localOut);
                        int offset = 0;
                        offset += xOffset;
                        offset += yOffset *(inputWidth - kernelSize + 1);
                        offset += (inputWidth - kernelSize + 1) * (inputHeight - kernelSize + 1) * outChan;
                        for (int sample = 0; sample < input.getWidth(); sample++) {
                            out.setValue(sample, offset, localOut.getValue(sample,0));
                        }
                    }
                }
            }

        }


        return out;
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        Collection<Matrix> free = new LinkedList<>();
        for (FullyConnectedLayer[] layerRow:
             layers) {
            for (FullyConnectedLayer layer: layerRow) {
                free.addAll(layer.getFreeVariables());
            }
        }
        return free;
    }

    public Matrix getKernelInput(Matrix m, int x, int y, int channel) {
        Matrix out = new Matrix(m.getWidth(), kernelSize*kernelSize);
        for (int cx = 0; cx < kernelSize; cx++) {
            for (int cy = 0; cy < kernelSize; cy++) {
                for (int sample = 0; sample < m.getWidth(); sample++) {
                    out.setValue(sample, cx + cy * kernelSize,  m.getValue(sample, x + cx + (y + cy) * inputWidth + channel * inputWidth * inputHeight));
                }
            }
        }
        return out;
    }


    public static void main(String[] args) throws InvalidDimensionsException {
        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new ConvolutionLayer(1, 2, 2, 2, 1));

        Matrix in = new Matrix(new float[][]{
                new float[]{1,0,0,0},
                new float[]{0,1,0,0},
                new float[]{0,0,1,0},
                new float[]{0,0,0,1},
        });

        System.out.println(nn.forwardPass(in).getHeight());
    }
}

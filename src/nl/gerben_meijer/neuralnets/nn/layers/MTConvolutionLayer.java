package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.mulithreading.Job;
import nl.gerben_meijer.neuralnets.mulithreading.ThreadPool;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Created by gerben on 6-1-17.
 * A layer consisting of multiple separated channels, each with their own fc neural network.
 * All subnets receive the same input and have the same output size.
 * The total output matrix has a height of output size times the number of channels
 */
public class MTConvolutionLayer extends Layer{

    private int outputChannels;
    private int kernelSize;
    private int inputWidth;
    private int inputHeight;
    private int inputChannels;
    private FullyConnectedLayer[][] layers;
    private ThreadPool pool = new ThreadPool();

    public MTConvolutionLayer(int outputChannels, int kernelSize, int inputWidth, int inputHeight, int inputChannels) {
        this.outputChannels = outputChannels;
        this.kernelSize = kernelSize;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputChannels = inputChannels;

        layers = new FullyConnectedLayer[inputChannels][outputChannels];
        for (int i = 0; i < inputChannels; i++) {
            for (int o = 0; o < outputChannels; o++) {
                layers[i][o] = new FullyConnectedLayer(kernelSize*kernelSize, 1, FullyConnectedLayer.InitOption.GAUSSIAN);
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

        LinkedList<Future<Boolean>> outs = new LinkedList<>();
        for (int xOffset = 0; xOffset < inputWidth - kernelSize + 1; xOffset++) {
            Collection<Job> jobs = new LinkedList<>();
            for (int yOffset = 0; yOffset < inputHeight - kernelSize+ 1; yOffset++) {
                ConvJob j = new ConvJob(input, xOffset, yOffset, out);
                outs.add(j.getOutput());
                jobs.add(j);
            }
            pool.addJobs(jobs);

        }

        for(Future<Boolean> m: outs) {
            try {
                m.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }


        return out;
    }

    @Override
    public Matrix backPropagate(Matrix input, Matrix error) throws InvalidDimensionsException {
        throw new NotImplementedException();
    }

    @Override
    public void updateVars(Matrix input, Matrix error) {

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

        nn.addLayer(new MTConvolutionLayer(1, 2, 2, 2, 1));

        Matrix in = new Matrix(new float[][]{
                new float[]{1,0,0,0},
                new float[]{0,1,0,0},
                new float[]{0,0,1,0},
                new float[]{0,0,0,1},
        });

        System.out.println(nn.forwardPass(in));
    }


    public class ConvJob extends Job<Boolean> {
        Matrix input; int xOffset; int yOffset; Matrix out;

        public ConvJob(Matrix input, int xOffset, int yOffset, Matrix out) {
            this.input = input;
            this.xOffset = xOffset;
            this.yOffset = yOffset;
            this.out = out;
        }

        @Override
        protected Boolean run() {
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
            return true;
        }
    }
}

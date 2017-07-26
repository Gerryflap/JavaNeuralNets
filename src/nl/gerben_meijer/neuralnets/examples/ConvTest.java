package nl.gerben_meijer.neuralnets.examples;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.math.functions.ReLU;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.math.optimize.GerbenOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.IMMLROptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.Optimizer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.ConvolutionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;

import java.util.Random;

/**
 * Created by gerben on 26-7-17.
 */
public class ConvTest {

    public static void main(String[] args) throws InvalidDimensionsException {
        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new ConvolutionLayer(3, 6, 10,10, 1));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));

        nn.addLayer(new ConvolutionLayer(6, 4, 5,5, 3));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));

        nn.addLayer(new FullyConnectedLayer(6*2*2, 5));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));

        nn.addLayer(new FullyConnectedLayer(5, 2));


        Matrix testY = new Matrix(new float[][]{
                new float[]{3.0f, 1.0f, 7.0f, 6.0f},
                new float[]{1.0f, 3.0f, 7.0f, 6.0f}
        });
        Matrix testX = generateImages(testY, 10, 10);

        System.out.println(testX);

        Matrix costY = generateRandomXYs(10, 10,10);
        Matrix costX = generateImages(costY, 10, 10);

        CostFunction f = new SoftmaxRateCostFunction();

        float lr = 0.3f;
        GerbenOptimizer optimizer = new GerbenOptimizer(lr, nn, f);

        for (int i = 0; i < 300; i++) {
            if (i!=0 && i%10 == 0) {
                lr/=1.5f;
                optimizer.setLearningRate(lr);
                System.out.println(nn.forwardPass(testX));
            }
            Matrix batchCorrect = generateRandomXYs(20, 10,10);
            Matrix batchX = generateImages(batchCorrect, 10, 10);

            optimizer.optimize(batchX, batchCorrect);

            System.out.printf("Iter: %d, Cost: %f\n", i, f.apply(nn.forwardPass(costX), costY));

        }

        System.out.println(nn.forwardPass(testX));
    }

    public static Matrix generateImages(Matrix positions, int width, int height) {
        Matrix out = new Matrix(positions.getWidth(), width * height);
        for (int i = 0; i < positions.getWidth(); i++) {
            out.setValue(i, (int) (positions.getValue(i, 0) + positions.getValue(i, 1) * width), 1);
        }
        return out;
    }

    public static Matrix generateRandomXYs(int n, int width, int height) {
        Matrix out = new Matrix(n, 2);
        Random random = new Random();
        return out.mapFunction(x -> random.nextInt(width));


    }
}

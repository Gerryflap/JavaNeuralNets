package nl.gerben_meijer.neuralnets.examples;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.MultiThreadMatrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.math.functions.Sigmoid;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.math.optimize.IMMROptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.MultilearnRateOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.Optimizer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import nl.gerben_meijer.neuralnets.nn.layers.Softmax;

/**
 * Created by gerben on 25-12-16.
 */
public class XorExample {

    public static void main(String[] args) throws InvalidDimensionsException {
        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new FullyConnectedLayer(2, 3));
        nn.addLayer(new ActivationFunctionLayer(new Sigmoid()));

        nn.addLayer(new FullyConnectedLayer(3, 2));
        nn.addLayer(new Softmax());

        float[][] inputData = new float[][]{
                new float[]{0,0,1,1},
                new float[]{0,1,0,1}
        };
        MultiThreadMatrix input = new MultiThreadMatrix(inputData);

        float[][] correctData = new float[][]{
                new float[]{0,1,1,0},
                new float[]{1,0,0,1},

        };
        MultiThreadMatrix correct = new MultiThreadMatrix(correctData);


        CostFunction costFunction = new SoftmaxRateCostFunction();
        Optimizer optimizer = new IMMROptimizer(0.2f, 0.00001f, nn, costFunction);
        float cost = (float) costFunction.apply(nn.forwardPass(input), correct);
        while (cost > 0.0000000001) {
            optimizer.optimize(input, correct);
            cost = (float) costFunction.apply(nn.forwardPass(input), correct);
            System.out.printf("Cost: %f\n", cost);
        }
        System.out.println(nn.forwardPass(input));
    }
}

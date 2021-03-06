package nl.gerben_meijer.neuralnets.examples;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.MultiThreadMatrix;
import nl.gerben_meijer.neuralnets.math.functions.*;
import nl.gerben_meijer.neuralnets.math.optimize.IMMLROptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.Optimizer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;

/**
 * Created by gerben on 25-12-16.
 */
public class XorExample {

    public static void main(String[] args) throws InvalidDimensionsException {
        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new FullyConnectedLayer(2, 4));
        nn.addLayer(new ActivationFunctionLayer(new TanH()));

        nn.addLayer(new FullyConnectedLayer(4, 1));
        nn.addLayer(new ActivationFunctionLayer(new TanH()));


        float[][] inputData = new float[][]{
                new float[]{0,0,1,1},
                new float[]{0,1,0,1}
        };
        MultiThreadMatrix input = new MultiThreadMatrix(inputData);

        float[][] correctData = new float[][]{
                new float[]{0,1,1,0},

        };
        MultiThreadMatrix correct = new MultiThreadMatrix(correctData);


        CostFunction costFunction = new SoftmaxRateCostFunction();
        Optimizer optimizer = new IMMLROptimizer(0.2f, 0.00001f, nn, costFunction);
        float cost = (float) costFunction.apply(nn.forwardPass(input), correct);
        int runs = 0;
        while (cost > 0.0000000001 && runs < 10000) {
            runs += 1;
            optimizer.optimize(input, correct);
            cost = (float) costFunction.apply(nn.forwardPass(input), correct);
            System.out.printf("Cost: %f\n", cost);
        }
        System.out.println(nn.forwardPass(input));
        System.out.println(runs);
    }
}

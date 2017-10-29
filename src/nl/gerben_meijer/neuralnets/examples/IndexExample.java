package nl.gerben_meijer.neuralnets.examples;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.MultiThreadMatrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.math.functions.Sigmoid;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.math.optimize.*;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

/**
 * Created by gerben on 23-12-16.
 * Test application
 */
public class IndexExample {

    public static void main(String[] args) throws InvalidDimensionsException {

        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new FullyConnectedLayer(50, 3));
        nn.addLayer(new ActivationFunctionLayer(new Sigmoid()));


        nn.addLayer(new FullyConnectedLayer(3, 1));


        float[][] inputData = new float[50][50];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i][i] = 1;
        }

        Matrix input = new MultiThreadMatrix(inputData);

        float[][] correctOutput = new float[1][50];
        for (int i = 0; i < 50; i++) {
            correctOutput[0][i] = i;
        }
        Matrix correct = new MultiThreadMatrix(correctOutput);

        CostFunction costFunction = new SoftmaxRateCostFunction();

        Optimizer optimizer = new IMMLROptimizer(0.05f, 0.00001f, nn, costFunction);
        float cost = (float) costFunction.apply(nn.forwardPass(input), correct);
        while (cost > 1) {
            optimizer.optimize(input, correct);
            cost = (float) costFunction.apply(nn.forwardPass(input), correct);
            System.out.println(cost);
            //System.out.println(optimizer.getLearningRates()[0]);
        }

        /**
        optimizer.setLearningRate(0.01f);

        while (cost > 1) {
            optimizer.optimize(input, correct);
            cost = (float) costFunction.apply(nn.forwardPass(input), correct);
            System.out.println(cost);
        }
         **/


        System.out.println(nn.forwardPass(input));

    }
}

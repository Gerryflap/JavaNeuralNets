package nl.gerben_meijer.neuralnets;

import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.MultiThreadMatrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.math.functions.Sigmoid;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.math.optimize.GerbenOptimizer;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.MultiplyLayer;
import nl.gerben_meijer.neuralnets.nn.layers.Softmax;

/**
 * Created by gerben on 23-12-16.
 * Test application
 */
public class TestApp {

    public static void main(String[] args) throws Matrix.InvalidDimensionsException {

        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new FullyConnectedLayer(100, 20));
        nn.addLayer(new ActivationFunctionLayer(new Sigmoid()));

        nn.addLayer(new FullyConnectedLayer(20, 1));


        float[][] inputData = new float[100][100];
        for (int i = 0; i < inputData.length; i++) {
            inputData[i][i] = 1;
        }

        Matrix input = new MultiThreadMatrix(inputData);

        float[][] correctOutput = new float[1][100];
        for (int i = 0; i < 100; i++) {
            correctOutput[0][i] = i;
        }
        Matrix correct = new MultiThreadMatrix(correctOutput);

        CostFunction costFunction = (output, correct1) -> {
            Matrix diff = output.add(correct1.mapFunction(x -> -x)).mapFunction(x -> (float) Math.pow(x,2));
            double total = 0.0;
            for (int depth = 0; depth < diff.getWidth(); depth++) {
                double cost = 0.0;
                for (int i = 0; i < diff.getHeight(); i++) {
                    cost += diff.getValue(depth, i);
                }
                total += cost;
            }

            return total;
        };

        GerbenOptimizer optimizer = new GerbenOptimizer(0.1f, nn, costFunction);
        float cost = (float) costFunction.apply(nn.forwardPass(input), correct);
        while (cost > 20) {
            optimizer.optimize(input, correct);
            cost = (float) costFunction.apply(nn.forwardPass(input), correct);
            System.out.println(cost);
        }

        optimizer.setLearningRate(0.01f);

        while (cost > 1) {
            optimizer.optimize(input, correct);
            cost = (float) costFunction.apply(nn.forwardPass(input), correct);
            System.out.println(cost);
        }


        System.out.println(nn.forwardPass(input));

    }
}

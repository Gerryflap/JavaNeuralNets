package nl.gerben_meijer.neuralnets;

import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.Sigmoid;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.Softmax;

/**
 * Created by gerben on 23-12-16.
 * Test application
 */
public class TestApp {

    public static void main(String[] args) throws Matrix.InvalidDimensionsException {

        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new FullyConnectedLayer(2, 10));
        nn.addLayer(new ActivationFunctionLayer(new Sigmoid()));

        nn.addLayer(new FullyConnectedLayer(10, 15));
        nn.addLayer(new ActivationFunctionLayer(new Sigmoid()));

        nn.addLayer(new FullyConnectedLayer(15, 2));
        nn.addLayer(new Softmax());


        Matrix output = nn.forwardPass(
                new Matrix(new float[][]{
                        new float[]{1},
                        new float[]{-2}
                })
        );

        System.out.println(output);
        System.out.println(new SoftmaxRateCostFunction().apply(output,
                new Matrix(new float[][]{
                        new float[]{0.f},
                        new float[]{1.f}
                })
                ));

    }
}

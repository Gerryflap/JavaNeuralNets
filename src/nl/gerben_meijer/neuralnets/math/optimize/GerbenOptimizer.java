package nl.gerben_meijer.neuralnets.math.optimize;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

import java.util.Collection;

/**
 * Created by gerben on 23-12-16.
 */
public class GerbenOptimizer extends Optimizer{

    private float learningRate;

    public GerbenOptimizer(float learningRate, NeuralNetwork neuralNetwork, CostFunction costFunction) {
        this.learningRate = learningRate;
        this.neuralNetwork = neuralNetwork;
        this.costFunction = costFunction;
    }

    public void optimizeNN(Matrix inputBatch, Matrix correctBatch) throws InvalidDimensionsException {

        for (Matrix m :
                neuralNetwork.getFreeVariables()) {
            for (int x = 0; x < m.getWidth(); x++) {
                for (int y = 0; y < m.getHeight(); y++) {
                    float lowestCost = Float.MAX_VALUE;
                    float lowestValue = 0.0f;

                    float normal = m.getValue(x, y);

                    for (int i = -1; i < 2; i++) {
                        float value = normal + i * learningRate;
                        m.setValue(x, y, value);
                        float cost = rateNetwork(inputBatch, correctBatch);
                        if (cost < lowestCost) {
                            lowestCost = cost;
                            lowestValue = value;
                        }
                        m.setValue(x, y, lowestValue);
                    }
                }
            }
        }
    }

    private float rateNetwork(Matrix input, Matrix correct) throws InvalidDimensionsException {
        return (float) costFunction.apply(neuralNetwork.forwardPass(input), correct);
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getLearningRate() {
        return learningRate;
    }
}

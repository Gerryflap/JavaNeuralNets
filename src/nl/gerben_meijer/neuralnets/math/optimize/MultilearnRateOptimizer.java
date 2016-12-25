package nl.gerben_meijer.neuralnets.math.optimize;

import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

/**
 * Created by gerben on 23-12-16.
 */
public class MultilearnRateOptimizer {

    private float learningRate;
    private NeuralNetwork neuralNetwork;
    private CostFunction costFunction;

    public MultilearnRateOptimizer(float learningRate, NeuralNetwork neuralNetwork, CostFunction costFunction) {
        this.learningRate = learningRate;
        this.neuralNetwork = neuralNetwork;
        this.costFunction = costFunction;
    }

    public void optimize(Matrix inputBatch, Matrix correctBatch) throws Matrix.InvalidDimensionsException {

        for (Matrix m :
                neuralNetwork.getFreeVariables()) {
            for (int x = 0; x < m.getWidth(); x++) {
                for (int y = 0; y < m.getHeight(); y++) {
                    float lowestCost = Float.MAX_VALUE;
                    float lowestValue = 0.0f;

                    float normal = m.getValue(x, y);

                    for (int i = -1; i < 2; i++) {
                        for (int expon = -2; i==0?expon < -1 : expon < 1; expon++) {
                            float pow = (float) Math.pow(10, expon);
                            float value = normal + i * pow * learningRate;
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
    }

    private float rateNetwork(Matrix input, Matrix correct) throws Matrix.InvalidDimensionsException {
        return (float) costFunction.apply(neuralNetwork.forwardPass(input), correct);
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }
}

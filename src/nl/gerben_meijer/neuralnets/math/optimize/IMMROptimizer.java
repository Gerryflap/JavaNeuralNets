package nl.gerben_meijer.neuralnets.math.optimize;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

import java.util.Iterator;

/**
 * Created by gerben on 23-12-16.
 * Individual Momentum MultilearnRate Optimizer
 *
 * Has an individual learnrate for every weight.
 */
public class IMMROptimizer implements Optimizer{

    private Matrix[] learningRates;
    private NeuralNetwork neuralNetwork;
    private CostFunction costFunction;
    private double maxLearningRate;
    private double minLearningRate;

    public IMMROptimizer(float maxLearningRate, float minLearningRate, NeuralNetwork neuralNetwork, CostFunction costFunction) {
        this.learningRates = new Matrix[neuralNetwork.getFreeVariables().size()];
        Iterator<Matrix> iterator = neuralNetwork.getFreeVariables().iterator();
        for (int i = 0; i < neuralNetwork.getFreeVariables().size(); i++) {
            Matrix m = iterator.next();
            learningRates[i] = new Matrix(m.getWidth(), m.getHeight()).mapFunction(x-> maxLearningRate);
        }
        this.neuralNetwork = neuralNetwork;
        this.costFunction = costFunction;
        this.maxLearningRate = maxLearningRate;
        this.minLearningRate = minLearningRate;
    }

    public void optimize(Matrix inputBatch, Matrix correctBatch) throws InvalidDimensionsException {

        Iterator<Matrix> iterator = neuralNetwork.getFreeVariables().iterator();
        for (int index = 0; index < neuralNetwork.getFreeVariables().size(); index++) {
            Matrix m = iterator.next();
            for (int x = 0; x < m.getWidth(); x++) {
                for (int y = 0; y < m.getHeight(); y++) {
                    float lowestCost = Float.MAX_VALUE;
                    float lowestValue = 0.0f;

                    float normal = m.getValue(x, y);
                    int lowestExpon = 0;
                    for (int i = -1; i < 2; i++) {

                        for (int expon = -1; i==0?expon < 0 : expon < 2; expon++) {
                            float pow = (float) Math.pow(2, expon);
                            float value = normal + i * pow * learningRates[index].getValue(x, y);
                            m.setValue(x, y, value);
                            float cost = rateNetwork(inputBatch, correctBatch);
                            if (cost < lowestCost) {
                                lowestCost = cost;
                                lowestValue = value;
                                if (i != 0) {
                                    lowestExpon = expon;
                                }
                            }

                        }

                    }
                    m.setValue(x, y, lowestValue);
                    learningRates[index].setValue(x, y,
                            (float) Math.max(minLearningRate, Math.min(maxLearningRate,
                                    (learningRates[index].getValue(x, y)* Math.pow(2, lowestExpon))
                            ))
                    );
                }
            }
        }


    }

    private float rateNetwork(Matrix input, Matrix correct) throws InvalidDimensionsException {
        return (float) costFunction.apply(neuralNetwork.forwardPass(input), correct);
    }

    public Matrix[] getLearningRates() {
        return learningRates;
    }
}

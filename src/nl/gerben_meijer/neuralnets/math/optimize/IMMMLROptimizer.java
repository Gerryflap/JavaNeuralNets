package nl.gerben_meijer.neuralnets.math.optimize;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.NetworkInput;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

import java.util.Iterator;

/**
 * Created by gerben on 23-12-16.
 * Individual Momentum Moving Multi LearnRate Optimizer
 *
 * Has an individual learnrate for every weight.
 */
public class IMMMLROptimizer extends Optimizer{

    private Matrix[] learningRates;
    private Matrix[] momentum;
    private double maxLearningRate;
    private double minLearningRate;
    private double momentumFactor;

    public IMMMLROptimizer(float maxLearningRate, float minLearningRate, float momentumFactor, NeuralNetwork neuralNetwork, CostFunction costFunction) {
        this.learningRates = new Matrix[neuralNetwork.getFreeVariables().size()];
        this.momentum = new Matrix[neuralNetwork.getFreeVariables().size()];

        Iterator<Matrix> iterator = neuralNetwork.getFreeVariables().iterator();
        for (int i = 0; i < neuralNetwork.getFreeVariables().size(); i++) {
            Matrix m = iterator.next();
            learningRates[i] = new Matrix(m.getWidth(), m.getHeight()).mapFunction(x-> maxLearningRate);
            momentum[i] = new Matrix(m.getWidth(), m.getHeight());
        }
        this.neuralNetwork = neuralNetwork;
        this.costFunction = costFunction;
        this.maxLearningRate = maxLearningRate;
        this.minLearningRate = minLearningRate;
        this.momentumFactor = momentumFactor;
    }

    public void optimizeNN(NetworkInput inputBatch, NetworkInput correctBatch) throws InvalidDimensionsException {

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
                            float pow = (float) Math.pow(1.1, expon);
                            float value = normal + (i * pow + momentum[index].getValue(x,y)) * learningRates[index].getValue(x, y);
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
                    momentum[index].setValue(x,y,
                            (float) ((1-momentumFactor) * momentum[index].getValue(x,y) + momentumFactor * (lowestValue-normal)));
                    learningRates[index].setValue(x, y,
                            (float) Math.max(minLearningRate, Math.min(maxLearningRate,
                                    (learningRates[index].getValue(x, y)* Math.pow(1.1, lowestExpon))
                            ))
                    );
                }
            }
        }


    }


    public Matrix[] getLearningRates() {
        return learningRates;
    }
}

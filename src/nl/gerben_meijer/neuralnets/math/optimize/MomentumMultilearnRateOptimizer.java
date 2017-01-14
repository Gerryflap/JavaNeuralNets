package nl.gerben_meijer.neuralnets.math.optimize;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

/**
 * Created by gerben on 23-12-16.
 */
public class MomentumMultilearnRateOptimizer implements Optimizer {

    private float learningRate;
    private NeuralNetwork neuralNetwork;
    private CostFunction costFunction;
    private int[] learnRateCount = new int[3];
    private double originalLearningrate;

    public MomentumMultilearnRateOptimizer(float learningRate, NeuralNetwork neuralNetwork, CostFunction costFunction) {
        this.learningRate = learningRate;
        this.neuralNetwork = neuralNetwork;
        this.costFunction = costFunction;
        this.originalLearningrate = learningRate;
    }

    public void optimize(Matrix inputBatch, Matrix correctBatch) throws InvalidDimensionsException {

        for (Matrix m :
                neuralNetwork.getFreeVariables()) {
            for (int x = 0; x < m.getWidth(); x++) {
                for (int y = 0; y < m.getHeight(); y++) {
                    float lowestCost = Float.MAX_VALUE;
                    float lowestValue = 0.0f;

                    float normal = m.getValue(x, y);
                    int lowestExpon = 0;
                    for (int i = -1; i < 2; i++) {

                        for (int expon = -1; i==0?expon < 0 : expon < 2; expon++) {
                            float pow = (float) Math.pow(2, expon);
                            float value = normal + i * pow * learningRate;
                            m.setValue(x, y, value);
                            float cost = rateNetwork(inputBatch, correctBatch);
                            if (cost < lowestCost) {
                                lowestCost = cost;
                                lowestValue = value;
                                lowestExpon = expon;
                            }

                        }


                    }
                    m.setValue(x, y, lowestValue);
                    learnRateCount[lowestExpon + 1] += 1;
                }
            }
        }


        int bestRate = 1;
        int highestUses = 0;
        for (int i = 0; i < 3; i++) {
            if (highestUses < learnRateCount[i]) {
                bestRate = i;
                highestUses = learnRateCount[i];
            }
        }

        //System.out.printf("Rates: %d, %d, %d. Bestrate = %d\n", learnRateCount[0], learnRateCount[1], learnRateCount[2], bestRate);
        double newLearningRate = learningRate * Math.pow(2, bestRate-1);
        if (newLearningRate <= originalLearningrate) {
            learningRate = (float) newLearningRate;
        }
        learnRateCount = new int[3];
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

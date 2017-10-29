package nl.gerben_meijer.neuralnets.math.optimize;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.NetworkInput;
import nl.gerben_meijer.neuralnets.math.Sequence;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

import java.util.ArrayList;

/**
 * Created by gerben on 14-1-17.
 *
 * Models an Optimizer
 */
public abstract class Optimizer {
    CostFunction costFunction;
    NeuralNetwork neuralNetwork;

    ArrayList<OptimizerListener> listeners = new ArrayList<>();

    public void addListener(OptimizerListener listener) {
        listeners.add(listener);
    }

    public void removeListener(OptimizerListener listener) {
        listeners.remove(listener);
    }

    public void optimize(Matrix inputBatch, Matrix correctBatch)  throws InvalidDimensionsException {
        optimizeNN(inputBatch, correctBatch);
        double cost = costFunction.apply(neuralNetwork.forwardPass(inputBatch), correctBatch);
        for (OptimizerListener listener: listeners) {
            listener.onOptimize(cost);
        }

    }

    public void optimize(Sequence inputSequence, Sequence correctSequence) throws InvalidDimensionsException {
        optimizeNN(inputSequence, correctSequence);
        double cost = costFunction.apply(neuralNetwork.forwardPass(inputSequence), correctSequence);
        for (OptimizerListener listener: listeners) {
            listener.onOptimize(cost);
        }
    }

    public abstract void optimizeNN(NetworkInput inputBatch, NetworkInput correctBatch) throws InvalidDimensionsException;

    protected float rateNetwork(NetworkInput input, NetworkInput correct) throws InvalidDimensionsException {
        return (float) costFunction.apply(neuralNetwork.forwardPass(input), correct);
    }
}

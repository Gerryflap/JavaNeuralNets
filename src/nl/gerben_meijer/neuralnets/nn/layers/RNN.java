package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.Sequence;
import nl.gerben_meijer.neuralnets.math.functions.Function;
import nl.gerben_meijer.neuralnets.math.functions.TanH;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * Created by gerben on 27-7-17.
 */
public class RNN extends Layer {
    private NeuralNetwork outputNN;
    private NeuralNetwork stateNN;
    private int inputSize;
    private int outputSize;
    private int stateSize;

    public RNN(int inputSize, int outputSize, int stateSize, Function outputActivation) {
        outputNN = new NeuralNetwork();
        stateNN = new NeuralNetwork();

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.stateSize = stateSize;

        outputNN.addLayer(new FullyConnectedLayer(inputSize + stateSize, outputSize));
        outputNN.addLayer(new ActivationFunctionLayer(outputActivation));

        stateNN.addLayer(new FullyConnectedLayer(inputSize + stateSize, stateSize, 0));
        stateNN.addLayer(new ActivationFunctionLayer(new TanH()));
    }

    @Override
    public Matrix forwardPass(Matrix input) {
        try {
            return outputNN.forwardPass(input.addAsRows(new Matrix(input.getWidth(), stateSize)));
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
            return null;
        }
    }

    public Sequence forwardPass(Sequence input) {
        Matrix state = new Matrix(input.getBatchSize(), stateSize);
        Iterator<Matrix> iterator = input.getIterator();
        Sequence output = new Sequence();
        while(iterator.hasNext()) {
            Matrix m = iterator.next();
            Matrix currentIn = null;
            try {
                currentIn = m.addAsRows(state);
                output.addMatrix(outputNN.forwardPass(currentIn));
                state = state.add(stateNN.forwardPass(currentIn));
            } catch (InvalidDimensionsException e) {
                e.printStackTrace();
            }
        }
        return output;
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        LinkedList<Matrix> freeVars = new LinkedList<>();
        freeVars.addAll(stateNN.getFreeVariables());
        freeVars.addAll(outputNN.getFreeVariables());
        return freeVars;
    }
}

package nl.gerben_meijer.neuralnets.nn;

import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.nn.layers.Layer;

import java.io.Serializable;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by gerben on 23-12-16.
 * Models a neural network
 */
public class NeuralNetwork implements Serializable{
    private List<Layer> layers = new LinkedList<>();

    public NeuralNetwork() {

    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public Matrix forwardPass(Matrix input) {
        for (Layer layer: layers) {
            input = layer.forwardPass(input);
        }
        return input;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public Collection<Matrix> getFreeVariables() {
        Collection<Matrix> collection = new LinkedList<>();
        for (Layer layer: layers) {
            collection.addAll(layer.getFreeVariables());
        }
        return collection;
    }

}

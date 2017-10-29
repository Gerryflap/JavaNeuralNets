package nl.gerben_meijer.neuralnets.nn.layers;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.Function;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;

/**
 * Created by gerben on 23-12-16.
 * The softmax function
 */
public class Softmax extends Layer{
    @Override
    public Matrix forwardPass(Matrix input) {
        input = input.mapFunction(x -> (float) Math.exp(x));
        float[][] data = new float[input.getHeight()][input.getWidth()];
        for (int x = 0; x < input.getWidth(); x++) {

            float sum = 0.0f;

            for (int y = 0; y < input.getHeight(); y++) {
                sum += input.getValue(x, y);
            }


            for (int y = 0; y < input.getHeight(); y++) {
                data[y][x] = input.getValue(x, y)/sum;
            }
        }
        try {
            return new Matrix(data);
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public Collection<Matrix> getFreeVariables() {
        return new LinkedList<>();
    }
}

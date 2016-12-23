package nl.gerben_meijer.neuralnets.math.functions;

import nl.gerben_meijer.neuralnets.math.Matrix;

/**
 * Created by gerben on 23-12-16.
 */
public class SoftmaxRateCostFunction implements CostFunction {


    @Override
    public double apply(Matrix output, Matrix correct) throws Matrix.InvalidDimensionsException {

        float[] ratings = new float[output.getWidth()];

        // Create a squared delta matrix
        Matrix diff = output.add(correct.mapFunction(x -> -x)).mapFunction(x -> (float) Math.pow(x,2));

        for (int x = 0; x < ratings.length; x++) {
            float cost = 0.0f;
            for (int y = 0; y < output.getHeight(); y++) {
                cost += diff.getValue(x, y);
            }
            ratings[x] = cost;
        }


        float sum = 0.0f;
        for (int i = 0; i < ratings.length; i++) {
            sum += ratings[i];
        }

        return sum/ratings.length;
    }
}

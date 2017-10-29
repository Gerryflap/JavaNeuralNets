package nl.gerben_meijer.neuralnets.examples;

import javafx.scene.paint.Material;
import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.Sequence;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.math.functions.ReLU;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.math.functions.TanH;
import nl.gerben_meijer.neuralnets.math.optimize.GerbenOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.IMMLROptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.MovingMultiLearnRateOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.Optimizer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import nl.gerben_meijer.neuralnets.nn.layers.RNN;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

import java.util.Iterator;

/**
 * Created by gerben on 27-7-17.
 */
public class RNNTest {
    public static void main(String[] args) throws InvalidDimensionsException {
        NeuralNetwork nn = new NeuralNetwork();
        nn.addLayer(new FullyConnectedLayer(1, 10));
        nn.addLayer(new ActivationFunctionLayer(new TanH()));

        nn.addLayer(new RNN(10, 10, 1, new TanH()));

        nn.addLayer(new FullyConnectedLayer(10, 10));
        nn.addLayer(new ActivationFunctionLayer(new TanH()));

        nn.addLayer(new FullyConnectedLayer(10, 1, FullyConnectedLayer.InitOption.GAUSSIAN));
        nn.addLayer(new ActivationFunctionLayer(x->Math.round(10.0*x)/10.0f));

        CostFunction costFunction = new SoftmaxRateCostFunction();
        GerbenOptimizer optimizer = new GerbenOptimizer(0.05f, nn, costFunction);

        Sequence testX = generateSequence(5, 10, 2.0f);
        Sequence testY = generateValidOuts(testX);

        Sequence printX = generateSequence(10, 1, 2.0f);
        Sequence printY = generateValidOuts(printX);

        for (int i = 0; i < 1000; i++) {
            Sequence in = generateSequence(5, 10, 2.0f);
            optimizer.optimize(in, generateValidOuts(in));
            System.out.printf("Iter: %d, cost: %f\n", i, costFunction.apply(nn.forwardPass(testX), testY));
            if (i%10 == 0) {
                System.out.println(printY);
                System.out.println(nn.forwardPass(printX));
                //optimizer.setLearningRate(optimizer.getLearningRate()/1.3f);
            }
        }

    }

    public static Sequence generateSequence(int n, int batchSize, float mult) {
        Sequence out = new Sequence();
        for (int i = 0; i < n; i++) {
            Matrix m = Matrix.initRandom(batchSize, 1, mult).mapFunction(x->Math.round(10.0*x)/10.0f);
            out.addMatrix(m);
        }
        return out;
    }

    public static Sequence generateValidOuts(Sequence inputs) throws InvalidDimensionsException {
        Sequence out = new Sequence();
        Iterator<Matrix> iterator = inputs.getIterator();
        Matrix totals = new Matrix(inputs.getBatchSize(), 1);
        int i = 1;
        while(iterator.hasNext()) {
            Matrix in = iterator.next();
            totals = totals.add(in);
            int finalI = i;
            //out.addMatrix(totals.mapFunction(x->x/ (float) finalI));
            out.addMatrix(totals);
            i++;
        }
        return out;

    }
}

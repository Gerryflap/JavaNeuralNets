package nl.gerben_meijer.neuralnets.examples;


import nl.gerben_meijer.neuralnets.math.optimize.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.CostFunction;
import nl.gerben_meijer.neuralnets.math.functions.ReLU;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.math.functions.TanH;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

/**
 * Created by gerben on 25-7-17.
 */
public class Sine extends ApplicationFrame{

    public Sine(String title) throws InvalidDimensionsException {
        super(title);
        JFreeChart xylineChart = run();
        ChartPanel chartPanel = new ChartPanel( xylineChart );
        chartPanel.setPreferredSize( new java.awt.Dimension( 560 , 367 ) );
        final XYPlot plot = xylineChart.getXYPlot( );

        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer( );
        renderer.setBaseShapesFilled(false);
        renderer.setDrawOutlines(false);
        plot.setRenderer( renderer );
        setContentPane( chartPanel );
    }

    public JFreeChart run() throws InvalidDimensionsException {
        NeuralNetwork nn = new NeuralNetwork();

        nn.addLayer(new FullyConnectedLayer(1, 50));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));

        nn.addLayer(new FullyConnectedLayer(50, 10));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));

        nn.addLayer(new FullyConnectedLayer(10, 3));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));


        nn.addLayer(new FullyConnectedLayer(3, 1));
        //nn.addLayer(new ActivationFunctionLayer(new TanH()));

        float[] testX_array = new float[200];
        float[] testY_array = new float[testX_array.length];

        for (int x = 0; x < testX_array.length; x++) {
            float xVal = (x/((float) testX_array.length));
            testX_array[x] = xVal;
            testY_array[x] = (float) Math.sin(20f*Math.PI*xVal);

        }

        Matrix testX = new Matrix(new float[][]{testX_array});
        Matrix testY = new Matrix(new float[][]{testY_array});

        CostFunction costFunction = new SoftmaxRateCostFunction();
        Optimizer optimizer = new MovingMultiLearnRateOptimizer(0.003f, nn, costFunction);

        for (int i = 0; i < 2750; i++) {
            Matrix input = Matrix.initRandom(32, 1, 1);
            Matrix output = input.mapFunction(x->(float) Math.sin(x*20f*Math.PI));
            //System.out.println(input);
            //System.out.println(output);
            optimizer.optimize(input, output);
            System.out.printf("Iter: %d, cost: %f\n" ,i ,costFunction.apply(nn.forwardPass(testX), testY));
        }




        XYSeriesCollection dataset = new XYSeriesCollection();
        XYSeries correct  = new XYSeries("Correct");
        XYSeries predicted = new XYSeries("Prediction");
        Matrix pred = nn.forwardPass(testX);
        for (int i = 0; i < testX.getWidth(); i++) {
            correct.add(testX.getValue(i,0), testY.getValue(i,0));
            predicted.add(testX.getValue(i,0), pred.getValue(i,0));
        }

        dataset.addSeries(correct);
        dataset.addSeries(predicted);

        JFreeChart chart = ChartFactory.createXYLineChart("Banaan", "Value", "X", dataset, PlotOrientation.VERTICAL ,
                true , true , false);

        System.out.println("Pred: ");
        System.out.println(nn.forwardPass(testX));
        System.out.println("Correct: ");
        System.out.println(testY);

        return chart;
    }

    public static void main(String[] args) throws InvalidDimensionsException {
        Sine sine = new Sine("Banaan");
        sine.pack();
        sine.setVisible(true);
    }
}

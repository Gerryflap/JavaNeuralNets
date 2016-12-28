package nl.gerben_meijer.neuralnets.examples;

import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.Sigmoid;
import nl.gerben_meijer.neuralnets.math.functions.SoftmaxRateCostFunction;
import nl.gerben_meijer.neuralnets.math.optimize.GerbenOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.MultilearnRateOptimizer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import nl.gerben_meijer.neuralnets.nn.layers.Softmax;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by gerben on 27-12-16.
 */
public class TicTacToe {

    float[] data = new float[9];
    ArrayList<float[][]> p1moves = new ArrayList<>();
    ArrayList<float[][]> p2moves = new ArrayList<>();

    public static void main(String[] args) {
        NeuralNetwork nn1 = new NeuralNetwork();
        nn1.addLayer(new FullyConnectedLayer(9, 15));
        nn1.addLayer(new ActivationFunctionLayer(new Sigmoid()));

        nn1.addLayer(new FullyConnectedLayer(15, 9));
        nn1.addLayer(new Softmax());

        NeuralNetwork nn2 = new NeuralNetwork();
        nn2.addLayer(new FullyConnectedLayer(9, 15));
        nn2.addLayer(new ActivationFunctionLayer(new Sigmoid()));

        nn2.addLayer(new FullyConnectedLayer(15, 9));
        nn2.addLayer(new Softmax());

        GerbenOptimizer optimizer1 = new GerbenOptimizer(0.01f, nn1, new SoftmaxRateCostFunction());
        GerbenOptimizer optimizer2 = new GerbenOptimizer(0.01f, nn2, new SoftmaxRateCostFunction());

        Random random = new Random();

        float wins = 0;
        float games = 0;
        while (true) {
            TicTacToe ticTacToe = new TicTacToe();
            int starting = random.nextInt(2) + 1;
            int player = starting;
            //System.out.printf("Starting player = %d\n", player);

            boolean rnd = true;//random.nextBoolean();

            while (ticTacToe.getWinner() == 0 && !ticTacToe.isFull()) {
                Matrix output;
                if (player == 1) {
                    output = nn1.forwardPass(ticTacToe.getBoard());
                } else {
                    if (!rnd) {
                        output = nn2.forwardPass(ticTacToe.getBoard());
                    } else {
                        output = Matrix.initRandom(1, 9);
                    }
                }

                float max = -1000;
                int pos = 0;

                for (int i = 0; i < 9; i++) {
                    float v = (float) (output.getValue(0, i) + random.nextGaussian() / 10.0);
                    //System.out.printf("%3f  ", v);
                    if (max < v && ticTacToe.isFree(i)) {
                        max = v;
                        pos = i;
                    }
                }

                //System.out.println();

                ticTacToe.makeMove(player, pos);

                player = player % 2 + 1;
            }

            int winner = ticTacToe.getWinner();


            if (ticTacToe.isFull()) {
                winner = starting%2+1;
            }

            games = (float) (1 + 0.99*games);

            if (winner == 1) {
                wins = (float) (1 + 0.99 * wins);
            } else {
                wins = (float) (0 + 0.99 * wins);
            }

            //System.out.printf("Winner = %d\n", winner);
            //System.out.println(ticTacToe);
            System.out.printf("Win percentage: %f %%\n", 100.0f*wins/(float) games);

            try {
                Matrix[] winning = ticTacToe.getTrainingMatrices(winner);
                optimizer1.optimize(winning[0], winning[1]);
                optimizer2.optimize(winning[0], winning[1]);

            } catch (Matrix.InvalidDimensionsException e) {
                e.printStackTrace();
            }


        }
    }

    private boolean isFree(int i) {
        return data[i] == 0;
    }

    public boolean makeMove(int player, int pos) {
        if (data[pos] != 0) {
            return false;
        } else {
            data[pos] = player;

            if (player == 1) {
                p1moves.add(new float[][] {data.clone(), getCorrectOutput(pos)});
            } else {
                p2moves.add(new float[][] {data.clone(), getCorrectOutput(pos)});
            }
            return true;
        }
    }

    public float[] getCorrectOutput(int pos) {
        float[] out = new float[9];
        out[pos] = 1;
        return out;
    }

    public int getWinner() {
        for (int x = 0; x < 3; x++) {
            int winner = (int) data[x];
            for (int y = 1; y < 3; y++) {
                int p = x + y*3;
                if (data[p] != winner || winner == 0) {
                    winner = 0;
                    break;
                }
            }
            if (winner != 0) {
                //System.out.println("Vertical win " + x);
                return winner;
            }
        }

        for (int y = 0; y < 3; y++) {
            int winner = (int) data[y*3];
            for (int x = 1; x < 3; x++) {
                int p = x + y*3;
                if (data[p] != winner || winner == 0) {
                    winner = 0;
                    break;
                }
            }
            if (winner != 0) {
                //System.out.println("Horizontal win" + y);
                return winner;
            }
        }

        int winner = (int) data[0];
        if (winner == data[4] && winner == data[8] && winner != 0) {
            //System.out.println("Diagonal win ls");
            return winner;
        }

        winner = (int) data[2];
        if (winner == data[4] && winner == data[6] && winner != 0) {
            //System.out.println("Diagonal win rs");
            return winner;
        }

        return 0;


    }

    public Matrix[] getTrainingMatrices(int player) throws Matrix.InvalidDimensionsException {
        if (player == 0) {
            return null;
        }

        ArrayList<float[][]> trainDataList = player == 1?p1moves:p2moves;

        float[][] inputs = new float[9][trainDataList.size()];
        float[][] correct = new float[9][trainDataList.size()];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = trainDataList.get(0)[0];
            correct[i] = trainDataList.get(0)[1];
        }

        Matrix[] out = new Matrix[]{
                new Matrix(inputs).transpose(),
                new Matrix(correct).transpose()
        };
        return out;
    }

    @Override
    public String toString() {
        String out = "";

        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                int p = x + y*3;
                out += String.format("%d  ", (int) data[p]);
            }
            out += "\n";
        }
        return out;
    }

    public Matrix getBoard() {
        Matrix board = null;
        try {
            board = new Matrix(new float[][]{data}).transpose();
        } catch (Matrix.InvalidDimensionsException e) {
            e.printStackTrace();
        }

        return board;
    }

    public boolean isFull() {
        for (int i = 0; i < 9; i++) {
            if (data[i] == 0) {
                return false;
            }
        }
        return true;
    }
}

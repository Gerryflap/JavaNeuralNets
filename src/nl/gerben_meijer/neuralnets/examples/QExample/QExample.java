package nl.gerben_meijer.neuralnets.examples.QExample;

import nl.gerben_meijer.neuralnets.ai.qlearning.Action;
import nl.gerben_meijer.neuralnets.ai.qlearning.DeepQAgent;
import nl.gerben_meijer.neuralnets.ai.qlearning.QCostFunction;
import nl.gerben_meijer.neuralnets.ai.qlearning.State;
import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.ReLU;
import nl.gerben_meijer.neuralnets.math.functions.Sigmoid;
import nl.gerben_meijer.neuralnets.math.functions.TanH;
import nl.gerben_meijer.neuralnets.math.optimize.GerbenOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.IMMROptimizer;
import nl.gerben_meijer.neuralnets.mulithreading.ThreadPool;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;
import nl.gerben_meijer.neuralnets.nn.layers.Softmax;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by gerben on 13-1-17.
 * This example has a Q-Agent move over a 9x9 board. The agent can move off the board, but is discouraged to do so.
 * Reaching the center (4,4) will reward the Agent and place it on a random position on the board.
 */
public class QExample {
    public static final List<Action> POSSIBLE_ACTIONS = new ArrayList<Action>();
    public static final int SIZE = 5;
    static {
        POSSIBLE_ACTIONS.add(new QAction(1,0));
        POSSIBLE_ACTIONS.add(new QAction(-1,0));
        POSSIBLE_ACTIONS.add(new QAction(0,1));
        POSSIBLE_ACTIONS.add(new QAction(0,-1));
    }

    public static void main(String[] args) throws InvalidDimensionsException {


        NeuralNetwork nn = new NeuralNetwork();
        nn.addLayer(new FullyConnectedLayer(SIZE*SIZE,10));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));

        nn.addLayer(new FullyConnectedLayer(10,4));
        nn.addLayer(new ActivationFunctionLayer(new ReLU()));


        nn.addLayer(new FullyConnectedLayer(4,4));
        //nn.addLayer(new ActivationFunctionLayer(new TanH()));

        DeepQAgent agent = new DeepQAgent(nn,
                new IMMROptimizer(0.001f, 0.00001f, nn, new QCostFunction()),
                new QState(0,0, 0, 0), POSSIBLE_ACTIONS,
                0.8f,  0.1f);

        while (true) {
            QAction actionDone = (QAction) agent.chooseAction();
            //**
            QState agentState = (QState) agent.getState();
            System.out.printf("Agent chose (%d, %d), now at (%d, %d)\n",
                    actionDone.dx, actionDone.dy,
                    agentState.x, agentState.y);
            for (int y = 0; y < SIZE; y++) {
                for (int x = 0; x < SIZE; x++) {
                    QState state = new QState(x, y, 0, 0);
                    System.out.printf("%.2f\t", agent.getQ(state));
                }
                System.out.println();
            }

             //**/

        }


    }

}

package nl.gerben_meijer.neuralnets.examples.QExample;

import nl.gerben_meijer.neuralnets.ai.qlearning.Action;
import nl.gerben_meijer.neuralnets.ai.qlearning.DeepQAgent;
import nl.gerben_meijer.neuralnets.ai.qlearning.State;
import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.functions.Sigmoid;
import nl.gerben_meijer.neuralnets.mulithreading.ThreadPool;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;
import nl.gerben_meijer.neuralnets.nn.layers.ActivationFunctionLayer;
import nl.gerben_meijer.neuralnets.nn.layers.FullyConnectedLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by gerben on 13-1-17.
 * This example has a Q-Agent move over a 9x9 board. The agent can move off the board, but is discouraged to do so.
 * Reaching the center (4,4) will reward the Agent and place it on a random position on the board.
 */
public class QExample {
    public static final List<Action> POSSIBLE_ACTIONS = new ArrayList<Action>();
    static {
        POSSIBLE_ACTIONS.add(new QAction(1,0));
        POSSIBLE_ACTIONS.add(new QAction(-1,0));
        POSSIBLE_ACTIONS.add(new QAction(0,1));
        POSSIBLE_ACTIONS.add(new QAction(0,-1));
    }

    public static void main(String[] args) throws InvalidDimensionsException {


        NeuralNetwork nn = new NeuralNetwork();
        nn.addLayer(new FullyConnectedLayer(2,5, 0));
        nn.addLayer(new ActivationFunctionLayer(new Sigmoid()));


        nn.addLayer(new FullyConnectedLayer(5,4, 0));

        DeepQAgent agent = new DeepQAgent(nn, new QState(3,4, 0, 0), POSSIBLE_ACTIONS, 0.8f, 0.01f, 0.3f);

        while (true) {
            QAction actionDone = (QAction) agent.chooseAction();
            QState agentState = (QState) agent.getState();
            System.out.printf("Agent chose (%d, %d), now at (%d, %d)\n",
                    actionDone.dx, actionDone.dy,
                    agentState.x, agentState.y);
            for (int y = 0; y < 10; y++) {
                for (int x = 0; x < 10; x++) {
                    QState state = new QState(x, y, 0, 0);
                    System.out.printf("%f\t", agent.getQ(state));
                }
                System.out.println();
            }


        }


    }

}

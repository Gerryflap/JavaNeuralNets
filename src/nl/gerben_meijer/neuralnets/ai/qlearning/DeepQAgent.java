package nl.gerben_meijer.neuralnets.ai.qlearning;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.optimize.GerbenOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.MomentumMultilearnRateOptimizer;
import nl.gerben_meijer.neuralnets.math.optimize.MultilearnRateOptimizer;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

import java.util.*;

/**
 * Created by gerben on 13-1-17.
 * A neural network implementation of a Q-learning agent
 *
 * Q-learning works by modeling States and Actions.
 * Every action will result in a state transition which might have a certain reward.
 */
public class DeepQAgent {
    private float explorationChance;
    private NeuralNetwork neuralNetwork;
    private Random random = new Random();
    private State currentState;
    private Map<Action, Integer> possibleActions;
    private MomentumMultilearnRateOptimizer optimizer;
    private LinkedList<Matrix[]> replayMemory = new LinkedList<>();
    private QCostFunction costFunction = new QCostFunction();
    private float dropoff;

    public DeepQAgent(NeuralNetwork neuralNetwork, State initialState, List<Action> possibleActions,
                      float dropoff, float learnrate, float explorationChance) throws InvalidDimensionsException {
        currentState = initialState;
        this.possibleActions = new HashMap<>();
        for (int i = 0; i < possibleActions.size(); i++) {
            this.possibleActions.put(possibleActions.get(i), i);
        }
        this.dropoff = dropoff;
        this.explorationChance = explorationChance;

        this.neuralNetwork = neuralNetwork;
        optimizer = new MomentumMultilearnRateOptimizer(learnrate, neuralNetwork, costFunction);


        //Forward pass the current state to test the network dimensions
        Matrix input = new Matrix(new float[][]{initialState.toNetworkInput()}).transpose();
        Matrix output = neuralNetwork.forwardPass(input);

        if (output.getHeight() != possibleActions.size() && output.getWidth() != 1) {
            throw new InvalidDimensionsException(
                    String.format("Deep Q Agent expected a Neural network with output (%d, %d), got (%d, %d)\n",
                            possibleActions.size(), 1,
                            output.getHeight(), output.getWidth()
                            )
            );
        }
    }

    /**
     * Choose an action and teach the Agent
     * @return the action the network took
     */
    public Action chooseAction() {
        try {
            Matrix input = new Matrix(new float[][]{currentState.toNetworkInput()}).transpose();
            Matrix output = neuralNetwork.forwardPass(input);
            Action bestAction = null;
            float highestReward = -Float.MAX_VALUE;
            Matrix correct = new Matrix(output.getData().clone());

            if (random.nextFloat() < explorationChance) {
                bestAction = currentState.getPossibleActions().get(random.nextInt(currentState.getPossibleActions().size()));
            } else {

                for (Action action :
                        currentState.getPossibleActions()) {
                    int index = possibleActions.get(action);
                    float expectedReward = output.getValue(0, index);
                    if (expectedReward > highestReward) {
                        bestAction = action;
                        highestReward = expectedReward;
                    }
                    System.out.printf("Expected reward for %s: %f\n", action, expectedReward);

                }
            }
            State newState = currentState.applyAction(bestAction);
            float reward = bestAction.getReward() + newState.getReward();
            System.out.printf("Reward1: %f\n", reward);

            Matrix nnout = neuralNetwork.forwardPass(
                    new Matrix(new float[][]{newState.toNetworkInput()})
                            .transpose());
            //System.out.println(nnout);
            //System.out.println("Max: "+ nnout.max());

            //Get best move from new state:
            float max = -Float.MAX_VALUE;
            for (int i = 0; i < newState.getPossibleActions().size(); i++) {
                float out = nnout.getValue(0, possibleActions.get(newState.getPossibleActions().get(i)));
                if (out > max) {
                    max = out;
                }
            }

            reward += dropoff * max;
            System.out.printf("Reward2: %f\n", reward);
            correct.setValue(0, possibleActions.get(bestAction), reward);
            System.out.println(correct);

            replayMemory.add(new Matrix[]{input, correct});

            if (random.nextInt(100) < replayMemory.size()) {
                learnReplayRecord();
                replayMemory.clear();
            }

            currentState = currentState.applyAction(bestAction);
            return bestAction;
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
        }
        return null;

    }

    public void learnReplayRecord() {
        float[][] inpData = new float[replayMemory.size()][];
        float[][] correctData = new float[replayMemory.size()][];
        for (int i = 0; i < inpData.length; i++) {
            inpData[i] = replayMemory.get(i)[0].transpose().getData()[0];
            correctData[i] = replayMemory.get(i)[1].transpose().getData()[0];
        }
        try {
            for (int i = 0; i < 100; i++) {
                optimizer.optimize(new Matrix(inpData).transpose(), new Matrix(correctData).transpose());
            }

        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
        }

    }


    public State getState() {
        return currentState;
    }

    public float getQ(State state) {
        try {
            return neuralNetwork.forwardPass(new Matrix(new float[][]{state.toNetworkInput()}).transpose()).max();
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
        }
        return Float.MAX_VALUE;
    }
}

package nl.gerben_meijer.neuralnets.ai.qlearning;

import nl.gerben_meijer.neuralnets.math.InvalidDimensionsException;
import nl.gerben_meijer.neuralnets.math.Matrix;
import nl.gerben_meijer.neuralnets.math.optimize.*;
import nl.gerben_meijer.neuralnets.nn.NeuralNetwork;

import java.io.*;
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
    private Optimizer optimizer;
    private LinkedList<Matrix[]> replayMemory = new LinkedList<>();
    private float dropoff;

    public DeepQAgent(String nnPath, Optimizer optimizer, State initialState, List<Action> possibleActions,
                      float dropoff, float explorationChance) throws InvalidDimensionsException, IOException, ClassNotFoundException {

        ObjectInputStream ooi = new ObjectInputStream(new FileInputStream(nnPath));
        NeuralNetwork neuralNetwork = (NeuralNetwork) ooi.readObject();
        init(neuralNetwork, optimizer, initialState, possibleActions, dropoff, explorationChance);
    }

    public DeepQAgent(NeuralNetwork neuralNetwork, Optimizer optimizer, State initialState, List<Action> possibleActions,
                      float dropoff, float explorationChance) throws InvalidDimensionsException {
        init(neuralNetwork, optimizer, initialState, possibleActions, dropoff, explorationChance);

    }

    private void init(NeuralNetwork neuralNetwork, Optimizer optimizer, State initialState, List<Action> possibleActions,
                      float dropoff, float explorationChance) throws InvalidDimensionsException {
        currentState = initialState;
        this.possibleActions = new HashMap<>();
        for (int i = 0; i < possibleActions.size(); i++) {
            this.possibleActions.put(possibleActions.get(i), i);
        }
        this.dropoff = dropoff;
        this.explorationChance = explorationChance;

        this.neuralNetwork = neuralNetwork;
        this.optimizer = optimizer;


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

            if (random.nextFloat() < explorationChance) {
                bestAction = currentState.getPossibleActions().get(random.nextInt(currentState.getPossibleActions().size()));
            }

            State newState = currentState.applyAction(bestAction);
            float reward = bestAction.getReward() + newState.getReward();
            //System.out.printf("Reward1: %f\n", reward);

            Matrix nnout = neuralNetwork.forwardPass(
                    new Matrix(new float[][]{newState.toNetworkInput()})
                            .transpose());
            //System.out.println(nnout);
            System.out.println("Max: "+ nnout.max());

            //Get best move from new state:
            float max = -Float.MAX_VALUE;
            for (int i = 0; i < newState.getPossibleActions().size(); i++) {
                float out = nnout.getValue(0, possibleActions.get(newState.getPossibleActions().get(i)));
                if (out > max) {
                    max = out;
                }
            }

            // Collect values:

            double oldQValue = highestReward;
            double discountFactor = dropoff;
            double learnrate = 0.1;
            double futureQEstimate = max;

            // The Q-function:

            System.out.printf("Values: \treward: %f, \tfutureQEstimate: %f, \toldQValue: %f\n",
                    reward, futureQEstimate, oldQValue);

            double learnValue = reward + discountFactor * futureQEstimate;
            double delta = learnrate * (learnValue - oldQValue);
            float newQValue = (float) (oldQValue + delta);

            System.out.printf("Q-value update: %f\n", delta);
            correct.setValue(0, possibleActions.get(bestAction), newQValue);
            System.out.println(correct);

            replayMemory.add(new Matrix[]{input, correct});

            if (random.nextInt(30) < replayMemory.size()) {
                for (int i = 0; i < 10; i++) {
                    learnReplayRecord();
                }
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
            Matrix in = new Matrix(inpData).transpose();
            Matrix correct = new Matrix(correctData).transpose();
            for (int i = 0; i < 1; i++) {
                optimizer.optimize(in, correct);
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
            Matrix out = neuralNetwork.forwardPass(new Matrix(new float[][]{state.toNetworkInput()}).transpose());
            float max = -Float.MAX_VALUE;
            for (Action action :
                    state.getPossibleActions()) {
                float score = out.getValue(0, possibleActions.get(action));
                if (score > max) {
                    max = score;
                }
            }
            return max;
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
        }
        return Float.MAX_VALUE;
    }

    public void save(String path) throws IOException {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path));
        oos.writeObject(this.neuralNetwork);
        oos.close();
    }
}

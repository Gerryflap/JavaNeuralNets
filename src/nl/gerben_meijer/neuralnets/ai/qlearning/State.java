package nl.gerben_meijer.neuralnets.ai.qlearning;

import java.util.List;

/**
 * Created by gerben on 13-1-17.
 * Models a State in the Q-learning model
 */
public interface State {

    /**
     * Converts the state to a float[] for the the network
     * @return A float[] representing the state
     */
    float[] toNetworkInput();

    /**
     * Applies an action to the state and returns the resulting state
     * @param a The action to perform
     * @return the new state
     */
    State applyAction(Action a);

    /**
     * Get the reward for reaching this state
     */
    float getReward();

    /**
     * Lists all possible actions
     * @return list of all possible actions from this state
     */
    List<Action> getPossibleActions();
}

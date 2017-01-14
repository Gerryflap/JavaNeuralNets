package nl.gerben_meijer.neuralnets.ai.qlearning;

/**
 * Created by gerben on 13-1-17.
 * Models an action in the Q-learning model
 *
 * Actions can have a certain reward
 */
public interface Action {


    float getReward();
}

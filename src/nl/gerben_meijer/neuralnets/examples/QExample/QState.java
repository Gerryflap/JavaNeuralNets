package nl.gerben_meijer.neuralnets.examples.QExample;

import nl.gerben_meijer.neuralnets.ai.qlearning.Action;
import nl.gerben_meijer.neuralnets.ai.qlearning.State;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static nl.gerben_meijer.neuralnets.examples.QExample.QExample.SIZE;

/**
 * Created by gerben on 13-1-17.
 */
public class QState implements State {
    private final int dx;
    private final int dy;
    final int x;
    final int y;
    final static Random random = new Random();

    public QState(int x, int y, int dx, int dy) {
        this.x = x;
        this.y = y;
        this.dx = dx;
        this.dy = dy;
    }

    @Override
    public float[] toNetworkInput() {
        float[] out = new float[SIZE*SIZE];
        out[x+y*SIZE] = 1;

        //float[] out = new float[]{x,y};
        return out;
    }

    @Override
    public State applyAction(Action a) {
        if (x == SIZE/2 && y == SIZE/2) {
            return new QState(random.nextInt(SIZE),random.nextInt(SIZE), 0,0);
        }
        QAction action = (QAction) a;
        int x = this.x + action.dx;
        int y = this.y + action.dy;

        return new QState(x, y, ((QAction) a).dx, ((QAction) a).dy);
    }

    @Override
    public float getReward() {


        if (x == SIZE/2 && y == SIZE/2) {
            return 100f;

        } else if ((x == SIZE-1 || x == 0) && (y == SIZE-1 || y == 0)) {
            return -10f;
        } else {
            return 0;
        }
    }

    @Override
    public List<Action> getPossibleActions() {
        List<Action> actions = new ArrayList<>();
        for (int i = 0; i < QExample.POSSIBLE_ACTIONS.size(); i++) {
            QAction action = (QAction) QExample.POSSIBLE_ACTIONS.get(i);

            if (x + action.dx < SIZE && x + action.dx >= 0 && y + action.dy < SIZE && y + action.dy >= 0 &&
                    (this.dx != -action.dx || this.dy != -action.dy) ) {
                actions.add(action);
            }
        }

        return actions;
    }
}


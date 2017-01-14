package nl.gerben_meijer.neuralnets.examples.QExample;

import nl.gerben_meijer.neuralnets.ai.qlearning.Action;
import nl.gerben_meijer.neuralnets.ai.qlearning.State;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
        float[] out = new float[10*10];
        out[x+y*10] = 1;

        //float[] out = new float[]{x,y};
        return out;
    }

    @Override
    public State applyAction(Action a) {
        if (x == 4 && y == 4) {
            return new QState(random.nextInt(10),random.nextInt(10), 0,0);
        }
        QAction action = (QAction) a;
        int x = this.x + action.dx;
        int y = this.y + action.dy;

        return new QState(x, y, ((QAction) a).dx, ((QAction) a).dy);
    }

    @Override
    public float getReward() {


        if (x == 4 && y == 4) {
            return 100000f;

        } else if ((x == 9 || x == 0) && (y == 9 || y == 0)) {
            return -10000f;
        } else {
            return 0;
        }
    }

    @Override
    public List<Action> getPossibleActions() {
        List<Action> actions = new ArrayList<>();
        for (int i = 0; i < QExample.POSSIBLE_ACTIONS.size(); i++) {
            QAction action = (QAction) QExample.POSSIBLE_ACTIONS.get(i);

            if (x + action.dx <= 9 && x + action.dx >= 0 && y + action.dy <= 9 && y + action.dy >= 0 &&
                    (this.dx != -action.dx || this.dy != -action.dy) ) {
                actions.add(action);
            }
        }

        return actions;
    }
}


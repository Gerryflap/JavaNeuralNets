package nl.gerben_meijer.neuralnets.examples.QExample;

import nl.gerben_meijer.neuralnets.ai.qlearning.Action;

/**
 * Created by gerben on 13-1-17.
 */
public class QAction implements Action {
    int dx = 0;
    int dy = 0;

    public QAction(int dx, int dy) {
        this.dx = dx;
        this.dy = dy;
    }

    @Override
    public float getReward() {
        return -0.01f;
    }

    public String toString() {
        return String.format("<QAction %d, %d>", dx, dy);
    }
}

package nl.gerben_meijer.neuralnets.math;

import java.util.Iterator;
import java.util.LinkedList;

/**
 * Created by gerben on 27-7-17.
 */
public class Sequence implements NetworkInput {
    public LinkedList<Matrix> data = new LinkedList<>();

    public void addMatrix(Matrix m) {
        if (data.size() == 0 || data.get(0).sizeEquals(m)) {
            data.add(m);
        }
    }

    public Iterator<Matrix> getIterator() {
        return data.descendingIterator();
    }

    public Iterable<Matrix> getIterable(){return data;}

    public int size() {
        return data.size();
    }

    public int getBatchSize() {
        if (data.size() == 0) {
            return 0;
        } else {
            return data.get(0).getWidth();
        }
    }

    public Matrix get(int i) {
        return data.get(i);
    }

    public String toString() {
        StringBuilder out = new StringBuilder();
        for (Matrix m: data) {
            out.append(m.toString());
        }
        return out.toString();
    }

}

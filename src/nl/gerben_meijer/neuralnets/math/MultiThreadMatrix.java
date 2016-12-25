package nl.gerben_meijer.neuralnets.math;

import nl.gerben_meijer.neuralnets.mulithreading.Job;
import nl.gerben_meijer.neuralnets.mulithreading.ThreadPool;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

/**
 * Created by gerben on 24-12-16.
 */
public class MultiThreadMatrix extends Matrix {

    public MultiThreadMatrix(int width, int height) {
        super(width, height);
    }

    public MultiThreadMatrix(float[][] data) throws InvalidDimensionsException {
        super(data);
    }

    public Matrix matmul(Matrix m) throws InvalidDimensionsException {
        Matrix t = this.transpose();

        if (t.width != m.width) {
            throw new InvalidDimensionsException(
                    String.format("Unable to multiply matrices of sizes (%d, %d) and (%d, %d)",
                            this.width, this.height,
                            m.width, m.height)
            );
        }

        float[][] newData = new float[m.height][t.height];
        Collection<Job> jobs = new LinkedList<>();
        MatrixJob[] matrixJobs = new MatrixJob[m.height];
        for (int y = 0; y < m.height; y++) {
            MatrixJob job = new MatrixJob(t, m, y);
            jobs.add(job);
            matrixJobs[y] = job;
        }

        ThreadPool.getInstance().addJobs(jobs);

        for (int y = 0; y < m.height; y++) {
            try {
                newData[y] = matrixJobs[y].getOutput().get();
                //System.out.printf("Received value %f\n", newData[y][depth]);
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }


        return new MultiThreadMatrix(newData);
    }

    private class MatrixJob extends Job<float[]> {

        Matrix a;
        Matrix b;
        int y;

        public MatrixJob(Matrix a, Matrix b, int y) {
            this.a = a;
            this.b = b;
            this.y = y;
        }

        @Override
        protected float[] run() {
            float[] newVals = new float[a.getHeight()];
            for (int depth = 0; depth < a.height; depth++) {
                float newVal = 0;
                for (int i = 0; i < b.getWidth(); i++) {
                    newVal += a.getValue(i, depth) * b.getValue(i, y);
                }
                newVals[depth] = newVal;
            }
            return newVals;
        }
    }
}

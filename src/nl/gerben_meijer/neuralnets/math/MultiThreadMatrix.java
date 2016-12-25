package nl.gerben_meijer.neuralnets.math;

import nl.gerben_meijer.neuralnets.math.functions.Function;
import nl.gerben_meijer.neuralnets.mulithreading.Job;
import nl.gerben_meijer.neuralnets.mulithreading.ThreadPool;
import nl.gerben_meijer.neuralnets.nn.layers.MultiplyLayer;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;

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

    public MultiThreadMatrix matmul(Matrix m) throws InvalidDimensionsException {
        if (width < 10) {
            return (MultiThreadMatrix) super.matmul(m);
        }
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
        MatrixMultJob[] matrixJobs = new MatrixMultJob[m.height];
        for (int y = 0; y < m.height; y++) {
            MatrixMultJob job = new MatrixMultJob(t, m, y);
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

    @Override
    public MultiThreadMatrix add(Matrix m) throws InvalidDimensionsException {
        if (width < 10) {
            return (MultiThreadMatrix) super.add(m);
        }
        if (this.width != m.width || this.height != m.height) {
            throw new InvalidDimensionsException(String.format("Tried adding matrices (%d, %d) and (%d, %d)",
                    this.width, this.height,
                    m.width, m.height));
        }
        float[][] newData = new float[this.height][this.width];
        Collection<Job> jobs = new LinkedList<>();
        MatrixAddJob[] jobArray = new MatrixAddJob[height];


        for (int y = 0; y < height; y++) {
            MatrixAddJob job = new MatrixAddJob(this, m, y);
            jobs.add(job);
            jobArray[y] = job;
        }

        ThreadPool.getInstance().addJobs(jobs);

        for (int y = 0; y < this.height; y++) {
            try {
                newData[y] = jobArray[y].getOutput().get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }


        return new MultiThreadMatrix(newData);
    }

    public MultiThreadMatrix mapFunction(Function f) {
        if (width < 10) {
            return (MultiThreadMatrix) super.mapFunction(f);
        }
        float[][] newData = new float[this.height][this.width];
        Collection<Job> jobs = new LinkedList<>();
        MatrixMapJob[] mapJobs = new MatrixMapJob[height];

        for (int y = 0; y < height; y++) {
            MatrixMapJob mapJob = new MatrixMapJob(this, f, y);
            jobs.add(mapJob);
            mapJobs[y] = mapJob;
        }

        ThreadPool.getInstance().addJobs(jobs);

        for (int y = 0; y < height; y++) {
            try {
                newData[y] = mapJobs[y].getOutput().get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        try {
            return new MultiThreadMatrix(newData);
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public MultiThreadMatrix addAsVector(Matrix m) throws InvalidDimensionsException {
        // TODO: Implement a job for this

        if (1 != m.width || this.height != m.height) {
            throw new InvalidDimensionsException(String.format("Tried vec-adding matrices (%d, %d) and (%d, %d)",
                    this.width, this.height,
                    m.width, m.height));
        }

        float[][] newData = new float[this.height][this.width];

        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < this.height; j++) {
                newData[j][i] = data[j][i] + m.getValue(0,j);
            }
        }

        return new MultiThreadMatrix(newData);
    }

    private class MatrixMultJob extends Job<float[]> {

        Matrix a;
        Matrix b;
        int y;

        public MatrixMultJob(Matrix a, Matrix b, int y) {
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

    private class MatrixAddJob extends Job<float[]> {
        Matrix a;
        Matrix b;
        int y;

        public MatrixAddJob(Matrix a, Matrix b, int y) {
            this.a = a;
            this.b = b;
            this.y = y;
        }

        @Override
        protected float[] run() {
            float[] newVals = new float[a.getWidth()];
            for (int x = 0; x < a.width; x++) {
                newVals[x] = a.getValue(x, y) + b.getValue(x, y);
            }
            return newVals;
        }
    }

    private class MatrixMapJob extends Job<float[]> {
        Matrix a;
        Function f;
        int y;

        public MatrixMapJob(Matrix a, Function function, int y) {
            this.a = a;
            this.f = function;
            this.y = y;
        }

        @Override
        protected float[] run() {
            float[] newVals = new float[a.getWidth()];
            for (int x = 0; x < a.width; x++) {
                newVals[x] = f.apply(a.getValue(x, y));
            }
            return newVals;
        }
    }


}

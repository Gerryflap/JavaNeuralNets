package nl.gerben_meijer.neuralnets.math;

import nl.gerben_meijer.neuralnets.math.functions.Function;

import java.util.Random;

/**
 * Created by gerben on 23-12-16.
 * Models a Matrix of floats
 */
public class Matrix {

    private float[][] data;
    private int width;
    private int height;

    public static void main(String[] args) throws InvalidDimensionsException {
        Matrix m1 = new Matrix(new float[][]{
                new float[]{1,1},
                new float[]{4,-1},
                new float[]{3,8},
                new float[]{6,-1},
                new float[]{-1,1}
        });

        Matrix m2 = new Matrix(new float[][]{
                new float[]{1,1,1,1,1},
                new float[]{0,1,1,1,1}
        });

        System.out.println(m1.matmul(m2));
    }

    public Matrix(int width, int height) {
        data = new float[width][height];
        this.width = width;
        this.height = height;
    }

    public Matrix(float[][] data) throws InvalidDimensionsException {
        height = data.length;
        this.data = data;
        if (height == 0) {
            throw new InvalidDimensionsException();
        }
        width = data[0].length;
        for (int i = 1; i < height; i++) {
            if (width != data[i].length) {
                throw new InvalidDimensionsException();
            }
        }
    }

    public float getValue(int x, int y) {
        return data[y][x];
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
        for (int depth = 0; depth < t.height; depth++) {
            for (int y = 0; y < m.height; y++) {
                for (int i = 0; i < m.width; i++) {
                    newData[y][depth] += t.getValue(i, depth) * m.getValue(i, y);
                }
            }
        }
        return new Matrix(newData);
    }

    public Matrix add(Matrix m) throws InvalidDimensionsException {
        if (this.width != m.width || this.height != m.height) {
            throw new InvalidDimensionsException(String.format("Tried adding matrices (%d, %d) and (%d, %d)",
                    this.width, this.height,
                    m.width, m.height));
        }
        float[][] newData = new float[this.height][this.width];

        for (int i = 0; i < this.width; i++) {
            for (int j = 0; j < this.height; j++) {
                newData[j][i] = data[j][i] + m.getValue(i,j);
            }
        }

        return new Matrix(newData);
    }

    public Matrix mapFunction(Function f) {
        float[][] newData = new float[this.height][this.width];

        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                newData[i][j] = f.apply(data[i][j]);
            }
        }
        try {
            return new Matrix(newData);
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
            return null;
        }
    }

    public Matrix transpose() {
        float[][] newData = new float[width][height];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                newData[j][i] = data[i][j];
            }
        }
        try {
            return new Matrix(newData);
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
            return null;
        }
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public class InvalidDimensionsException extends Exception {
        public InvalidDimensionsException() {
        }

        public InvalidDimensionsException(String message) {
            super(message);
        }
    }

    public String toString() {
        String out = "Matrix: \n";
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                out += String.format("%f", data[y][x]).substring(0,5) + "   ";
            }
            out += "\n";
        }
        return out;
    }

    public static Matrix initRandom(int width, int height) {
        Random random = new Random();
        float[][] data = new float[height][width];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                data[j][i] = (float) random.nextGaussian();
            }
        }

        try {
            return new Matrix(data);
        } catch (InvalidDimensionsException e) {
            e.printStackTrace();
            return null;
        }
    }
}

package nl.gerben_meijer.neuralnets.math;

/**
 * Created by gerben on 13-1-17.
 */
public class InvalidDimensionsException extends Exception {
    public InvalidDimensionsException() {
    }

    public InvalidDimensionsException(String message) {
        super(message);
    }
}
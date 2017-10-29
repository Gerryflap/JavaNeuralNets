package nl.gerben_meijer.neuralnets.math.functions;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.Serializable;

/**
 * Created by gerben on 23-12-16.
 * Models a function
 */
public interface Function extends Serializable{

    float apply(float x);

}

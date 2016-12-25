package nl.gerben_meijer.neuralnets.mulithreading;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

/**
 * Created by gerben on 24-12-16.
 * Models a job
 */
public abstract class Job<OutputType> {

    public enum JobState {QUEUED, PROCESSING, DONE}

    private JobState status = JobState.QUEUED;
    CompletableFuture<OutputType> out = new CompletableFuture<OutputType>();

    public void start() throws IllegalJobStateException {
        synchronized (this){
            if (status == JobState.QUEUED) {
                status = JobState.PROCESSING;
            } else {
                throw new IllegalJobStateException();
            }
        }

        OutputType out = this.run();


        this.out.complete(out);
        synchronized (this){
            if (status == JobState.PROCESSING) {
                status = JobState.DONE;
            } else {
                throw new IllegalJobStateException();
            }

        }


    }

    public Future<OutputType> getOutput() {
        return out;
    }

    public JobState getStatus() {
        return status;
    }

    protected abstract OutputType run();


}

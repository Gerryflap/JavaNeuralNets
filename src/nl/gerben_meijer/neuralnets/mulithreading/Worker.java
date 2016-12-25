package nl.gerben_meijer.neuralnets.mulithreading;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Created by gerben on 24-12-16.
 */
public class Worker extends Thread{

    private ThreadPool threadPool;
    private CyclicBarrier barrier = new CyclicBarrier(2);
    private Job job;
    boolean free = true;

    public Worker(ThreadPool threadPool) {
        this.threadPool = threadPool;
    }

    @Override
    public void run() {

        while(true) {
            if (free) {
                try {
                    barrier.await();
                } catch (InterruptedException | BrokenBarrierException e) {
                    e.printStackTrace();
                    continue;
                }
            }
            try {
                job.start();
            } catch (IllegalJobStateException e) {
                e.printStackTrace();
            }
            free = threadPool.free(this);
        }
    }

    public void runJob(Job job, boolean restart) {
        this.job = job;

        if (!restart) {
            try {
                //System.out.println("Waiting for barrier");
                barrier.await();
                //System.out.println("Barrier released");
            } catch (InterruptedException | BrokenBarrierException e) {
                e.printStackTrace();
            }
        }
        free = false;
    }
}

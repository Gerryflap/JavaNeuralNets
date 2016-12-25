package nl.gerben_meijer.neuralnets.mulithreading;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by gerben on 24-12-16.
 */
public class ThreadPool {

    public static final int NUM_THREADS = 8;
    private final List<Worker> freeThreads = new LinkedList<>();
    private final List<Worker> busyThreads = new LinkedList<>();
    private final List<Job> jobs = new LinkedList<>();
    public static ThreadPool instance = null;


    public ThreadPool() {
        for (int i = 0; i < NUM_THREADS; i++) {
            freeThreads.add(new Worker(this));
        }

        for (Worker worker :
                freeThreads) {
            worker.start();
        }
    }

    public static ThreadPool getInstance() {
        if (instance == null) {
            instance = new ThreadPool();
        }
        return instance;
    }

    public void addJobs(Collection<Job> jobs) {
        synchronized (this.jobs) {
            this.jobs.addAll(jobs);
            synchronized (freeThreads) {
                while (freeThreads.size() > 0 && this.jobs.size() > 0) {
                    //System.out.printf("Adding job, %d jobs available \n", this.jobs.size());
                    freeThreads.get(0).runJob(this.jobs.get(0), false);
                    busyThreads.add(freeThreads.get(0));
                    freeThreads.remove(0);
                    this.jobs.remove(0);
                }
            }
        }
    }


    /**
     * Frees worker
     * @param worker the worker
     * @return whether there is a new job;
     */
    public boolean free(Worker worker) {

        synchronized (jobs) {
            if (jobs.size() > 0) {
                //System.out.printf("Adding job, %d jobs available \n", jobs.size());
                worker.runJob(jobs.get(0), true);
                jobs.remove(0);
                return false;
            } else {
                synchronized (freeThreads) {
                    freeThreads.add(worker);
                    busyThreads.remove(worker);
                }
                return true;
            }
        }

    }

}

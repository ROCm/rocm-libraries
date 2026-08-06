import create_test_instance
import subprocess
import benchmarker
import threading

qqcompiled_total = 0
qqcompiled_success = 0
qqcompiled_failed = 0
qqcompiled_lock = threading.Lock()

def compiler_thread(threadId, work_queue, benchmark_queue):
    global qqcompiled_total, qqcompiled_success, qqcompiled_failed
    while True:
        # try to get new work. Catch a shutdown
        try:
            my_replace_vals, file_idx, source, uid, test_instance_config = (
                work_queue.get()
            )
            if (
                my_replace_vals is None
            ):  # workaround for python <3.13, which do not have shutdown
                work_queue.task_done()
                return
        except Exception:  # queue is shutdown, no more work -> return
            return

        print("Thread %d starting compile" % (threadId))

        instance_file_name, result = create_test_instance.compile_test_instance(
            test_instance_config, my_replace_vals, threadId, file_idx
        )

        # Update global counters safely
        with qqcompiled_lock:
            qqcompiled_total += 1
            if result == 0:
                qqcompiled_success += 1
            else:
                qqcompiled_failed += 1
            print(f"[Compile Summary] total={qqcompiled_total}, success={qqcompiled_success}, failed={qqcompiled_failed}")

        # to avoid overloadind benchmark queue, put only if build succeeded, otherwise delete
        if result == benchmarker.SUCCESS:
            print("Thread %d finished compile res %d" % (threadId, result))
            benchmark_queue.put(
                [instance_file_name, result, my_replace_vals, source, uid]
            )
        elif not test_instance_config[
            "print_build_failures"
        ]:  # do not delete failed instances when in testing mode
            subprocess.getstatusoutput(
                "rm " + instance_file_name + " && rm " + instance_file_name + ".o"
            )
        work_queue.task_done()

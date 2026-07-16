import create_test_instance
import subprocess
import benchmarker


def compiler_thread(threadId, work_queue, benchmark_queue):
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

        instance_file_name, result = create_test_instance.compile_test_instance(
            test_instance_config, my_replace_vals, threadId, file_idx
        )
        # to avoid overloadind benchmark queue, put only if build succeeded, otherwise delete
        if result == benchmarker.SUCCESS:
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

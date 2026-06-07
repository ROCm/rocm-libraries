def sched_iter_alg_is_zero(schedule_iter_alg: int, tailloop_in_nll: bool, nll_last: bool) -> bool:
    """Mirror of KernelWriter._makeSubIterSchedule (KernelWriter.py:877-882) local
    `scheduleIterAlg`, then the predicate `scheduleIterAlg == 0` at line 882.

    schedule_iter_alg == self.states.scheduleIterAlg, which (KernelLanguage='Assembly')
      == kernel['_ScheduleIterAlg'] == the YAML ScheduleIterAlg for SIA in {0,1,3}.
    tailloop_in_nll == the param tailloopInNll == self.states.tailloopInNll == YAML TailloopInNll.
    nll_last == the param NLLlast (True unless PGR>=2 and isNGLL, set in noLoadLoop).

    Override (line 878-880): if (NLLlast and tailloopInNll): scheduleIterAlg = 0.
    Predicate (line 882): scheduleIterAlg == 0.

    pre: schedule_iter_alg in (0, 1, 3)
    post: __return__ == ((nll_last and tailloop_in_nll) or schedule_iter_alg == 0)
    """
    sched = schedule_iter_alg
    if nll_last and tailloop_in_nll:
        sched = 0
    return sched == 0
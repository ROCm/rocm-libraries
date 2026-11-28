<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor: weekly report and lessons review (tick step 5)

5. **Weekly report.** If the tick prints `WEEKLY REPORT <week> ready:` and a file's path, read that file and present it
   at the top of your reply:
   - the headline table;
   - every PR that broke develop, with its fixer;
   - the top friction rows;
   - one line on age and throughput;
   - the path.

   The first tick after Monday 14:00 UTC (`weekly_report.weekday` and `weekly_report.hour_utc`) writes the previous
   week's report, once; a week missed while the monitor was down is written late, oldest first.

   **Lessons review:** every week, distil the week's issues and findings into improvements to the agentic process, so no
   erroneous cycle repeats and nothing is re-discovered. The same tick prints `LESSONS REVIEW <week> due:` and the path
   of the week's raw material, `rocke_lessons_raw_<week>.md` in the weekly reports' folder (`paths.weekly_dir`;
   `rk lessons [--week YYYY-Www]` makes it on demand). Dispatch one process analyst (no `model` option: it runs on the
   session's model) with:
   - that file;
   - `lessons.md`;
   - the archived verifier notes;
   - the feedback memories that changed that week;
   - the drafters' records: `state/drafter_assessment.jsonl` (one record per drafter per review, from the verifier's
     lines with the live and the shadow drafters' tags; with no first review pass and no shadow drafter, none),
     `state/scaler.jsonl` (every draft start the scaler allowed or held, with its inputs), `state/review_cost.jsonl`
     (each review's cost: the first review pass, the shadow drafter, the verifier and the code quality pass) and the
     drafters' card (`GLM drafter vs Opus`: `rk drafts`);
   - last week's `guards_<week>.md` and the toolkit's commits of the week.

   It returns:
   - the week's erroneous cycles: count, cost, and whether a lesson already existed. A lesson that existed and failed
     becomes a toolkit guard or test. The evidence for that comes from the archived verifier notes
     (`<WORKING_DIR>/lessons/notes/`) and the week's raw material, not from `lessons.md`;
   - the drafter's week, from its records (when a drafter drafted): what it missed and claimed falsely, its tokens and
     minutes, its missing or late drafts, and what the scaler held back;
   - which of last week's proposed guards were built; a guard proposed in two reviews and still unbuilt goes to
     <user_login> with its weekly cost (L-206);
   - new and retired lessons;
   - the exact edits to `lessons.md`, the review book's §11 (the mirror of `lessons.md`'s Worker section), the
     verifier's team member file and this file.

   Lessons are written at face value, in the format `lessons.md`'s introduction gives: the rule in the present tense
   and the tool that enforces it, if one does. Their evidence, dates and PR examples stay in the archived notes and
   the week's `lessons_archive_<week>.md`, not in `lessons.md`.

   In the weekly reports' folder (`paths.weekly_dir`), the analyst first moves the previous week's outputs into a
   `w<NN>_final/` folder, then writes: `lessons_draft.md` (install it as `<LESSONS_FILE>` after a diff, keeping the old
   one as `.bak-YYYY-MM-DD`), `lessons_archive_<week>.md` (copy it to `<WORKING_DIR>/lessons/archive/`),
   `w<NN>_review.md`, `embodiment_edits.md` and `guards_<week>.md`. It runs `lessons_for.py --check` on its draft in a
   scratch copy (`skillcopy/.claude/skills/rocke-monitor/` and `skillcopy/.claude/team_members/`, the draft installed as
   `lessons.md`) and reports each role's size.

   Present the review in compact tables. Apply the lesson and mirror edits at once, build the proposed guards as toolkit
   code with tests, and ask <user_login> before changing a standing rule. Headless (a container): apply lesson edits to
   the lessons file only, and write team-file, review-book, skill and guard proposals to `<week>_proposals.md` in the
   weekly reports' folder, which <user_login> lands as a pull request to the package; the session never
   pushes, and the week's corrections come from the journal and the Needs-you answers, not from memories.

   After editing `lessons.md`, the verifier's or the code quality reviewer's team member file, or the review passes'
   commands card (`lane_commands.md`, kept in step with `rk --help`), run `python -B lessons_for.py --check`. It prints
   each role's extract size and fails on a part over 29,000 bytes (Bash shows an agent at most 30,000 characters at
   once), a cited lesson missing from `lessons.md` and the site lessons, a missing card, or site lessons it cannot read.
   Keep the verifier's extract to two parts, three with the site lessons: its team file prints `--part 1` and
   `--part 2` and follows a next-part line, and every part stays in each verifier's context for its whole run.
   Site-specific lessons (hosts, people, tickets, pools) go to the site lessons, `<SITE_LESSONS_FILE>`, never
   `lessons.md`.

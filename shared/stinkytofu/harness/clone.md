# Git in a sparse or partial clone

Read this when `git fetch` fails or prints an error, or when the clone is sparse or partial.

- If `git fetch` fails, tell the developer and continue. In a sparse or partial clone it can print
  `bad tree object` or `failed to run repack`. That message comes from background garbage collection,
  so check whether the fetch completed: `git rev-parse FETCH_HEAD` must resolve, and
  `git log -1 --format=%cd FETCH_HEAD` should be a recent date. If it is old, run the fetch again.
- Do not run `git grep` over history or the whole tree. It downloads missing objects one at a time and
  can hang. Search the working tree instead.

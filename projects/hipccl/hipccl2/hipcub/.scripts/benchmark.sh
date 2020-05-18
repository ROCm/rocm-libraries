 #!/bin/bash
for i in $(ls | egrep -i 'benchmark_*' ); do
  ./$i
done;

#!/usr/bin/env bash
set -euo pipefail

SRC_TEST="pth_II_rwl.c"
SRC_RAND="my_rand.c"
SRC_TASK="task3.c"      
OUT_PTHREAD="bench_pthread"
OUT_CUSTOM="bench_custom"
REPEATS=5              
THREAD_COUNTS=(1 2 4 8 16 32)
TOTAL_OPS=50000       
INSERTS_IN_MAIN=1000

WORKLOADS=(
  "read_heavy 0.90 0.09 0.01"
  "balanced   0.50 0.25 0.25"
  "write_heavy 0.20 0.40 0.40"
)

OUT_CSV="results2.csv"

echo "Compiling binaries..."
gcc -O2 -pthread "$SRC_TEST" "$SRC_RAND" -o "$OUT_PTHREAD" || { echo "Compile pthread failed"; exit 1; }
gcc -O2 -pthread -DUSE_CUSTOM "$SRC_TEST" "$SRC_RAND" "$SRC_TASK" -o "$OUT_CUSTOM" || { echo "Compile custom failed"; exit 1; }
echo "Compiled: $OUT_PTHREAD and $OUT_CUSTOM"

cat > "$OUT_CSV" <<'CSV'
binary,thread_count,workload,search,insert,delete,total_ops,inserts_in_main,repeat,elapsed_s,total_ops_reported,member_ops,insert_ops,delete_ops,throughput_ops_per_s
CSV

run_one() {
  local bin=$1
  local threads=$2
  local search=$3
  local insert=$4
  local delete=$5
  local total_ops=$6
  local inserts_in_main=$7
  local repeat_no=$8

  local out
  out=$(printf "%d\n%d\n%f\n%f\n" "$inserts_in_main" "$total_ops" "$search" "$insert" | ./"$bin" "$threads" 2>&1)

  local elapsed
  elapsed=$(echo "$out" | awk -F'=' '/Elapsed time/ {gsub(/ seconds/,"",$2); gsub(/ /,"",$2); print $2; exit}')
  local total_ops_reported
  total_ops_reported=$(echo "$out" | awk -F'=' '/Total ops/ {gsub(/ /,"",$2); print $2; exit}')
  local member_ops insert_ops delete_ops
  member_ops=$(echo "$out" | awk -F'=' '/member ops/ {gsub(/ /,"",$2); print $2; exit}')
  insert_ops=$(echo "$out" | awk -F'=' '/insert ops/ {gsub(/ /,"",$2); print $2; exit}')
  delete_ops=$(echo "$out" | awk -F'=' '/delete ops/ {gsub(/ /,"",$2); print $2; exit}')

  local throughput=0
  if [[ -n "$elapsed" && "$elapsed" != "0" ]]; then
    throughput=$(awk -v t="$total_ops_reported" -v s="$elapsed" 'BEGIN{printf("%.0f", t / s)}')
  fi

  printf "%s,%d,%s,%.6f,%.6f,%.6f,%d,%d,%d,%.6f,%s,%s,%s,%s,%s\n" \
    "$bin" "$threads" "$workload_label" "$search" "$insert" "$delete" "$total_ops" "$inserts_in_main" "$repeat_no" "$elapsed" "$total_ops_reported" "$member_ops" "$insert_ops" "$delete_ops" "$throughput"
}

echo "Starting benchmark runs..."
for workload_spec in "${WORKLOADS[@]}"; do
  read -r workload_label search_frac insert_frac delete_frac <<<"$workload_spec"

  for threads in "${THREAD_COUNTS[@]}"; do
    for ((r=1; r<=REPEATS; r++)); do
      echo "RUN: pthread binary, workload=$workload_label, threads=$threads, repeat=$r"
      run_one "$OUT_PTHREAD" "$threads" "$search_frac" "$insert_frac" "$delete_frac" "$TOTAL_OPS" "$INSERTS_IN_MAIN" "$r" >> "$OUT_CSV"

      echo "RUN: custom binary, workload=$workload_label, threads=$threads, repeat=$r"
      run_one "$OUT_CUSTOM" "$threads" "$search_frac" "$insert_frac" "$delete_frac" "$TOTAL_OPS" "$INSERTS_IN_MAIN" "$r" >> "$OUT_CSV"
    done
  done
done

echo "All runs complete. Results in $OUT_CSV"
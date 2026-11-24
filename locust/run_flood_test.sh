#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-http://127.0.0.1}"
REPLICAS="${REPLICAS:-1 2 4}"
USERS="${USERS:-10 50 100}"
SPAWN_RATE="${SPAWN_RATE:-10}"
DURATION="${DURATION:-3m}"
WARMUP="${WARMUP:-30s}"
REPEATS="${REPEATS:-1}"
OUT="${OUT:-locust/results}"
LOCUST="${LOCUST:-locust}"
PYTHON="${PYTHON:-python3}"

wait_for_api() {
    for _ in $(seq 1 60); do
        curl -fsS "$HOST/health" > /dev/null 2>&1 && return 0
        sleep 2
    done
    echo "The API at $HOST/health did not become healthy" >&2
    exit 1
}

scale_api() {
    echo "Scaling api to $1 containers"
    docker compose up -d --scale api="$1" --no-recreate api nginx
    wait_for_api
    sleep 15
}

run_locust() {
    "$LOCUST" -f locust/locustfile.py --headless --host "$HOST" -u "$1" -r "$SPAWN_RATE" -t "$2" --only-summary "${@:3}"
}

record_docker_stats() {
    sleep 60
    docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' > "$1"
}

measure() {
    local tag="$1" users="$2"
    echo "Run $tag: warm-up $WARMUP (discarded), then $DURATION"
    run_locust "$users" "$WARMUP" > /dev/null 2>&1 || true
    record_docker_stats "$OUT/${tag}_docker_stats.csv" &
    LOCUST_HOSTNAME_CSV="$OUT/${tag}_hostnames.csv" run_locust "$users" "$DURATION" --csv "$OUT/$tag" || true
    wait
}

main() {
    mkdir -p "$OUT"
    for replicas in $REPLICAS; do
        scale_api "$replicas"
        for users in $USERS; do
            for repeat in $(seq 1 "$REPEATS"); do
                measure "r${replicas}_u${users}_rep${repeat}" "$users"
            done
        done
    done
    "$PYTHON" locust/summarize.py "$OUT"
}

main

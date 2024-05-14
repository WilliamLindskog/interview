#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


for strategy in fedavg fedadagrad fedadam fedyogi; do
    echo "Running $strategy"
    python server.py --num_rounds 40 --strategy $strategy &
    sleep 3  # Sleep for 3s to give the server enough time to start

    for i in $(seq 0 4); do
        echo "Starting client $i"
        python client.py --partition-id "$i" --alpha 0.5 &
    done

    # Enable CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait
done

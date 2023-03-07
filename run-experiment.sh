#!/bin/bash

experiment_folder=$(pwd)/experiments/
experiment_name=$1

start_time=$(date +"%H:%M:%S")
screen -d -m -S "${experiment_name}" -L -Logfile "./${experiment_name}_${start_time}.log" nice -n0 python "${experiment_folder}${experiment_name}.py" & pids+=($!)
wait "${pids[@]}"



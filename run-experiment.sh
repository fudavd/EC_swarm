#!/bin/bash

experiment_folder=$(pwd)/experiments/
experiment_name=$1

date +"%H:%M:%S"
screen -d -m -S "${experiment_name}" -L -Logfile "./${experiment_name}.log" nice -n0 python "${experiment_folder}${experiment_name}.py" & pids+=($!)
wait "${pids[@]}"
date +"%H:%M:%S"


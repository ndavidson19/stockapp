#!/bin/bash
python /path/to/data_fetching_script.py &
echo $! > /tmp/data_fetching_pid.txt

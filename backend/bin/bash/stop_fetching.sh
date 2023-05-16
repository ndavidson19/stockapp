#!/bin/bash
kill $(cat /tmp/data_fetching_pid.txt)
rm /tmp/data_fetching_pid.txt

#!/bin/bash
# Save the data to a file
/path/to/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic stock_topic --from-beginning > /path/to/backup.txt
# Delete the data from the topic
/path/to/kafka-topics.sh --zookeeper localhost:2181 --delete --topic stock_topic
# Recreate the topic
/path/to/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic stock_topic
# Stop the data-fetching script
kill $(cat /tmp/data_fetching_pid.txt)
rm /tmp/data_fetching_pid.txt

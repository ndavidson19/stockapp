from confluent_kafka import Consumer, KafkaException

conf = {'bootstrap.servers': 'localhost:9092', 'group.id': 'mygroup', 'auto.offset.reset': 'earliest'}
consumer = Consumer(conf)

# Subscribe to topic
consumer.subscribe(['stock_topic'])

while True:
    msg = consumer.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        raise KafkaException(msg.error())
    else:
        # Proper message
        sys.stderr.write('%% %s [%d] at offset %d with key %s:\n' %
                        (msg.topic(), msg.partition(), msg.offset(),
                        str(msg.key())))
        print(msg.value())

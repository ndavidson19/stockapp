val spark = SparkSession.builder.appName("KafkaStreaming").getOrCreate()

val df = spark
  .readStream
  .format("kafka")
  .option("kafka.bootstrap.servers", "localhost:9092")
  .option("subscribe", "stock_topic")
  .load()


dataframe.write
  .format("jdbc")
  .option("url", "jdbc:mysql://localhost/db")
  .option("driver", "com.mysql.jdbc.Driver")
  .option("dbtable", "stock")
  .option("user", "username")
  .option("password", "password")
  .save()

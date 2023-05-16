import requests
import time
from confluent_kafka import Producer

def fetch_stock_data(symbol, api_key):
    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_INTRADAY'
    interval = '1min'

    data = {
        "function": function,
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key
    }

    response = requests.get(base_url, params=data)

    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    api_key = 'your_alphavantage_api_key'  # replace with your API key
    symbol = 'MSFT'  # replace with the stock symbol you're interested in

    # Kafka configuration
    kafka_conf = {'bootstrap.servers': 'localhost:9092'}
    producer = Producer(kafka_conf)

    while True:
        # Fetch data from AlphaVantage
        data = fetch_stock_data(symbol, api_key)

        if data is not None:
            # Publish data to Kafka
            producer.produce('stock_topic', value=str(data))
            producer.flush()

        # Sleep for 1 minute before fetching data again
        time.sleep(60)


if __name__ == '__main__':
    main()

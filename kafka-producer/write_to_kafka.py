from kafka import KafkaProducer
import json
import time
import schedule
import pandas as pd

time.sleep(10) # Wait for Kafka to start

# loading the dataset from a remote repository for reproducability 
# no not need to download and place it manually in the kafka-producer folder
dataset_url = 'https://huggingface.co/datasets/divakaivan/kb_project_data_to_stream/resolve/main/data_to_stream.csv'
streaming_df = pd.read_csv(dataset_url, index_col=0)

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'test'

current_index = 0

def publish_messages():
    global current_index
    
    rows = streaming_df.iloc[current_index:current_index + 5].to_dict(orient='records')
    for row in rows:
    
        producer.send(topic, row)
        print('Data inserted:', row)
    
    current_index += 5
    
    if current_index >= len(streaming_df):
        current_index = 0

if __name__ == '__main__':
    time.sleep(10)
    schedule.every(1).seconds.do(publish_messages)
    while True:
        schedule.run_pending()
        time.sleep(0.01)

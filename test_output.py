from kafka import KafkaConsumer

import json
import base64

class MessageConsumer:
    def __init__(self, broker, topic):
        self.broker = broker
        self.consumer = KafkaConsumer(
            topic,  # Topic to consume
            bootstrap_servers=self.broker,
            value_deserializer=lambda x: x.decode(
                "utf-8"
            ),  # Decode message value as utf-8
            group_id="my-group",  # Consumer group ID
            auto_offset_reset="earliest",  # Start consuming from earliest available message
            enable_auto_commit=True,  # Commit offsets automatically
        )

    def receive_message(self):
        try:
            for message in self.consumer:
                print(message.value)
                print(json.loads(message.value)["result"])
                
                print("\n")
                
                #print(json.dumps(message.value).decode('UTF-8'))
        except Exception as exc:
            raise exc


# 브로커와 토픽명을 지정한다.
broker = ["localhost:9092", "localhost:9093", "localhost:9094"]
topic = "LLM_output"
cs = MessageConsumer(broker, topic)
cs.receive_message()
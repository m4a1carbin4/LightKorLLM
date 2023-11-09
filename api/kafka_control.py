import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(
    os.path.abspath(os.path.dirname(__file__)))+"/lib")

from kafka import KafkaConsumer
from kafka import KafkaProducer
import json
import time
from Infer import Infer

class LLM_KafkaControl:
    def __init__(self,infer:Infer,broker,topics):
        self.infer = infer
        self.broker = broker
        self.pd_topic = topics[0]
        self.cs_topic = topics[1]
        self.producer = KafkaProducer(
            bootstrap_servers=self.broker,
            client_id="LLM",
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            acks=0,
            api_version=(3,6,0),
            retries=3,
        )
        self.consumer = KafkaConsumer(
            self.cs_topic,  # Topic to consume
            bootstrap_servers=self.broker,
            value_deserializer=lambda x: x.decode(
                "utf-8"
            ),  # Decode message value as utf-8
            group_id="LLM-group",  # Consumer group ID
            auto_offset_reset="earliest",  # Start consuming from earliest available message
            enable_auto_commit=True,  # Commit offsets automatically
        )
    
    def send_message(self,msg,auto_close=True):
        try:
            future = self.producer.send(self.pd_topic, msg)
            self.producer.flush()  # 비우는 작업
            if auto_close:
                self.producer.close()
            future.get(timeout=2)
            return {"status_code": 200, "error": None}
        except Exception as exc:
            raise exc
    
    def receive_message(self):
        try:
            for message in self.consumer:
                
                result_str, result_history = self.infer.text_gen(
                    data=json.loads(message.value), type="infer")
                
                self.send_message({
                    "result": result_str,
                    "history": result_history
                },False)
        except Exception as exc:
            raise exc
        
        
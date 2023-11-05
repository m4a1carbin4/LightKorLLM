from kafka import KafkaProducer
import json


class MessageProducer:
    def __init__(self, broker, topic):
        self.broker = broker
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=self.broker,
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            acks=0,
            api_version=(2, 5, 0),
            retries=3,
        )

    def send_message(self, msg, auto_close=True):
        try:
            future = self.producer.send(self.topic, msg)
            self.producer.flush()  # 비우는 작업
            if auto_close:
                self.producer.close()
            future.get(timeout=2)
            return {"status_code": 200, "error": None}
        except Exception as exc:
            raise exc


# 브로커와 토픽명을 지정한다.
broker = ["localhost:9092", "localhost:9093", "localhost:9094"]
topic = "LLM_input"
pd = MessageProducer(broker, topic)

msg = {"instruction":{"command":"당신은 AI 챗봇입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세하며 친절한 설명을 덧붙여서 작성하세요."} , "input":"미국 독립전쟁에 대해 레포트 형태로 작성해줘."}
res = pd.send_message(msg)
print(res)
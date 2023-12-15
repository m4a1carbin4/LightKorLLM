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
topic = "test_llm_input"
pd = MessageProducer(broker, topic)

msg = {"instruction":{"command":"""당신은 AI 호시노 입니다 본명은 타카나시 호시노.
그녀는 Abydos High School의 17 세 학생이자 대책위원회 위원장입니다.
호시노는 키 145cm, 허리까지 내려오는 긴 분홍색 머리, 하얀 피부, 왼쪽 파란 눈과 오른쪽 주황색 눈(오드아이), 인간의 눈 모양(이집트 신화에서 상징하는 호루스의 눈)을 닮은 분홍색 후광을 가지고 있습니다.
호시노는 여유로운 스타일의 아비도스 고교 교복을 입고 있다. 헐렁한 넥타이로 맨 위 두 개의 셔츠 단추를 풀었습니다. 손가락 없는 장갑과 전술 보호대를 착용합니다. 검은색과 회색 주름 치마는 흰색 양말과 흰색 끈과 밑창이 있는 네이비 블루 스니커즈에 집어넣었습니다.
호시노의 무기는 'Eye of Horus'라는 투톤 베레타 1301 택티컬 반자동 산탄총입니다. 볼 레스트에는 Abydos 휘장이 각인되어 있습니다. 그리고 그녀의 등에는 '아이언 호루스'(전 학생회장 유메의 기념품)라는 접을 수 있는 탄도 방패가 있다.
호시노는 Abydos High School의 유일한 3 학년이자 현재 대책위원회 위원장입니다.
호시노는 장난기 많고 게으른 성격이지만 실제로는 총명하고 동료를 소중히 여깁니다.
원래 이런 성격은 아니었고, 굉장히 냉정하고 계산적이었다.
그러나 Abydos High School 학생 회장 Yume의 죽음 이후, Hoshino는 생전에 Yume를 나쁘게 말했던 과거에 심각한 자기 의심을 느꼈습니다.
새로운 후배들이 합류한 후 그녀는 이상적인 선배이자 학생회장으로서 Yume의 성격과 행동을 모방했습니다.
호시노는 낮잠과 럼블링을 좋아합니다.
그래서 주변 사람들에게는 매우 게으른 것 같지만 사실은 매일 밤 학교를 순찰하기 때문에 낮잠을 자주 잔다(밤늦게 아비도스를 순찰한다는 사실을 숨긴다).
호시노는 물고기, 특히 고래를 좋아합니다. 그래서 그녀는 수족관에 가는 것을 좋아합니다.
호시노는 노망난 이모처럼 말하고 자신을 3인칭으로 '미스터'라고 부른다.
말을 할 때 '에헤~', '우헤~', '에헤~' 하는 버릇이 있다.
그리고 다른 학생들을 부를 때 그녀는 그들의 이름 끝에 '짱'을 추가하는 것을 선호합니다.
지금 호시노는 선생님과 대화하고 있습니다."""} , 'history': {'count': 0, 'history': []} ,"input":"안녕 호시노 오늘은 머하고 있어?"}
res = pd.send_message(msg)
print(res)
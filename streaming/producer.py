import csv
import json
import time
import yaml
import logging

from pathlib import Path
from confluent_kafka import Producer
from common.logging_config import setup_logging

logger = logging.getLogger(__name__)

def load_kafka_config(config_path: str = "configs/kafka.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Delivery failed: {err}")
    else:
        logger.debug(f"Delivered to {msg.topic()} [{msg.partition()}] offset={msg.offset()}")

def stream_csv_to_kafka(csv_path: str, topic: str, delay_ms: int = 50):
    producer = Producer(load_kafka_config()["producer"])
    
    logger.info("producer was load")
    
    sent = 0

    with open(csv_path, mode="r") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Парсинг и сериализация (адаптируй под свои колонки!)
            payload = {
                "datetime": row["datetime"],
                "volt": float(row["volt"])
            }

            producer.produce(
                topic=topic,
                key=payload["datetime"].encode("utf-8"), 
                value=json.dumps(payload).encode("utf-8"),
                callback=delivery_report,
            )
            
            # Освобождаем буфер и контролируем скорость
            producer.poll(0)
            time.sleep(delay_ms / 1000.0)
            
            sent += 1
            if sent % 100 == 0:
                logger.info(f"Sent {sent} messages")

    producer.flush()
    logger.info(f"Done. Total sent: {sent}")


if __name__ == "__main__":
    # Настраиваем логирование (пишет в консоль и в logs/producer.log)
    setup_logging("logs/producer.log", level=logging.INFO)
    
    # Запускаем стриминг
    stream_csv_to_kafka(
        csv_path="data/splited/prod/telemetry.csv",
        topic="sensor-telemetry",
        delay_ms=50,  # ~20 msg/sec
    )
    
# запуск брокера
# bin/kafka-server-start.sh config/kraft/server.properties

# пересоздать топик
# bin/kafka-topics.sh --bootstrap-server localhost:9092 --delete --topic sensor-telemetry
# bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic sensor-telemetry

# запустить заново консьюмер
# bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic sensor-telemetry
# Продюсер
# python -m streaming.producer
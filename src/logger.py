import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs', datetime.now().strftime('%d-%m-%Y'))

os.makedirs(name=logs_path, mode=0o777, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# if __name__ == "__main__":
#     logging.info("logging has started.")
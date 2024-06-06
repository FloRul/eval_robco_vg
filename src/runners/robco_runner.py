import time
from typing import Optional, Tuple
from fmeval.model_runners.model_runner import ModelRunner
import json
import logging

from utils import ThrottledWebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobcoRunner(ModelRunner):
    def __init__(self, ws_address: str, output_intent: bool = False, *args, **kwargs):
        assert ws_address is not None, "WebSocket address must be provided"
        self.ws_address = ws_address
        self.ws = None
        self.output_intent = output_intent
        self.connect()

    def connect(self):
        try:
            self.ws = ThrottledWebSocket(self.ws_address)
            logger.info(f"Connected to WebSocket at {self.ws_address}")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["ws"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.connect()

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        request_params = {
            "message": prompt,
        }
        request_json = json.dumps(request_params)

        try:
            self.ws.send(request_json)
            response_json = self.ws.recv()
            response_dict = json.loads(response_json)
            result = response_dict.get("message")
            intent = response_dict.get("intent")
            final = (
                result + f"<intention>{intent}</intention>"
                if self.output_intent
                else result
            )
            logger.info({final})
            return final, None
        except Exception as e:
            logger.warning("WebSocket timeout, retrying...")
            time.sleep(5)  # Wait before retrying
            self.connect()  # Reconnect the WebSocket
            try:
                self.ws.send(request_json)
                response_json = self.ws.recv()
                response_dict = json.loads(response_json)
                result = response_dict.get("message")
                logger.info(f"Received response after retry: {result}")
                return result, None
            except Exception as e:
                logger.error(f"Failed after retry: {e}")
                return None, None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None, None

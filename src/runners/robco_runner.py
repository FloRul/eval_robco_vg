import time
from typing import Optional, Tuple
from fmeval.model_runners.model_runner import ModelRunner
import json
import logging

from utils import ThrottledWebSocket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobcoRunner(ModelRunner):
    def __init__(
        self,
        ws_address: str,
        output_intent: bool = False,
        ws_origin: str = None,
    ):
        assert ws_address is not None, "WebSocket address must be provided"
        assert ws_origin is not None, "WebSocket origin must be provided"
        self.ws_address = ws_address
        self.ws_origin = ws_origin
        self.ws = None
        self.output_intent = output_intent
        self.connect()

    def connect(self):
        try:
            self.ws = ThrottledWebSocket(ws=self.ws_address, ws_origin=self.ws_origin)
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

    def fetch_model_output(self, prompt: str) -> Tuple[str, float]:
        self.ws.send(json.dumps({"message": prompt}))
        response_json = self.ws.recv()
        response_dict = json.loads(response_json)
        result = response_dict.get("message")
        intent = response_dict.get("intent")
        final = f"<intention>{intent}</intention>" if self.output_intent else result
        return final, None

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        try:
            return self.fetch_model_output(prompt)
        except Exception as e:
            try:
                logger.warning("WebSocket timeout, retrying...")
                time.sleep(2)  # Wait before retrying
                self.connect()  # Reconnect the WebSocket
                return self.fetch_model_output(prompt)
            except Exception as e:
                logger.error(f"Failed after retry: {e}")
                return (
                    "TestError: Une erreur est survenue dans le processus de test",
                    None,
                )

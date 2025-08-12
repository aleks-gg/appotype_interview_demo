import yaml
import threading
from pathlib import Path

class ServerConfig:
    """
    Singleton configuration class for the server settings.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    _settings = None

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ServerConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._load_config()
            self._initialized = True

    def _load_config(self):
        """Loads configuration from the `server_config.yaml` file located in the same directory as this file."""
        config_path = (Path(__file__).parent / 'server_config.yaml').absolute()

        try:
            with open(config_path, 'r') as f:
                self._settings = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing YAML file at {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load server configuration: {e}")

    @property
    def config_data(self):
        """Returns the loaded configuration data."""
        return self._settings

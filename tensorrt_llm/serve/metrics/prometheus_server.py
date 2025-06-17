from prometheus_client import start_http_server
import threading
import logging

logger = logging.getLogger(__name__)

class PrometheusServer:
    def __init__(self, port=8000):
        self.port = port
        self.server_thread = None

    def start(self):
        """Start Prometheus metrics server in a separate thread."""
        def run_server():
            try:
                start_http_server(self.port)
                logger.info(f"Prometheus metrics server started on port {self.port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

    def stop(self):
        """Stop the Prometheus metrics server."""
        if self.server_thread and self.server_thread.is_alive():
            # Note: prometheus_client doesn't provide a direct way to stop the server
            # The server will be stopped when the process exits
            logger.info("Prometheus metrics server will be stopped with the main process") 
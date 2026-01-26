"""
API server for industrial recommendation system.
Exposes recall and rank services via HTTP.
"""
import logging
from typing import Any, Dict, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from config import get_config
from serving.recall_service import RecallService
from serving.rank_service import RankService

logger = logging.getLogger(__name__)


class RecsysAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for recommendation API."""

    def log_message(self, format, *args):
        """Override to use logger."""
        logger.info(f"API: {args[0]}")

    def _send_json_response(self, status: int, data: Dict[str, Any]) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json_response(200, {
                "status": "healthy",
                "recall_service": self.server.recall_service.health_check(),
                "rank_service": self.server.rank_service.health_check(),
            })
        elif self.path == "/stats":
            self._send_json_response(200, {
                "status": "ok",
            })
        else:
            self._send_json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/recall":
            self._handle_recall()
        elif self.path == "/rank":
            self._handle_rank()
        elif self.path == "/recommend":
            self._handle_recommend()
        else:
            self._send_json_response(404, {"error": "Not found"})

    def _handle_recall(self) -> None:
        """Handle recall request."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            user_id = data.get("user_id")
            top_k = data.get("top_k", 100)

            if not user_id:
                self._send_json_response(400, {"error": "user_id is required"})
                return

            item_ids, scores = self.server.recall_service.recall(user_id, top_k=top_k)

            self._send_json_response(200, {
                "user_id": user_id,
                "item_ids": item_ids,
                "scores": scores,
                "count": len(item_ids),
            })
        except Exception as e:
            logger.error(f"Recall error: {e}")
            self._send_json_response(500, {"error": str(e)})

    def _handle_rank(self) -> None:
        """Handle rank request."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            user_id = data.get("user_id")
            item_ids = data.get("item_ids", [])

            if not user_id:
                self._send_json_response(400, {"error": "user_id is required"})
                return

            if not item_ids:
                self._send_json_response(400, {"error": "item_ids is required"})
                return

            ranked = self.server.rank_service.rank(user_id, item_ids)

            self._send_json_response(200, {
                "user_id": user_id,
                "ranked_items": ranked,
                "count": len(ranked),
            })
        except Exception as e:
            logger.error(f"Rank error: {e}")
            self._send_json_response(500, {"error": str(e)})

    def _handle_recommend(self) -> None:
        """Handle recommend request (recall + rank)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            user_id = data.get("user_id")
            top_k = data.get("top_k", 10)

            if not user_id:
                self._send_json_response(400, {"error": "user_id is required"})
                return

            # Step 1: Recall
            recall_item_ids, recall_scores = self.server.recall_service.recall(
                user_id, top_k=100
            )

            # Step 2: Rank
            ranked = self.server.rank_service.rank(user_id, recall_item_ids, top_k=top_k)

            self._send_json_response(200, {
                "user_id": user_id,
                "recommendations": ranked,
                "count": len(ranked),
            })
        except Exception as e:
            logger.error(f"Recommend error: {e}")
            self._send_json_response(500, {"error": str(e)})


class RecsysAPIServer(HTTPServer):
    """Recommendation API server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Initialize API server.

        Args:
            host: Host to bind
            port: Port to listen
        """
        super().__init__((host, port), RecsysAPIHandler)

        # Initialize services
        self.recall_service = RecallService()
        self.recall_service.initialize()

        self.rank_service = RankService()

        logger.info(f"API server initialized on {host}:{port}")


def start_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """
    Start the recommendation API server.

    Args:
        host: Host to bind
        port: Port to listen
    """
    server = RecsysAPIServer(host, port)
    logger.info(f"Starting API server on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    start_server(args.host, args.port)

"""
Online learning module for industrial recommendation system.
Handles real-time feature updates and model incremental updates.
"""
import logging
import threading
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class EventBuffer:
    """
    Ring buffer for streaming events.
    Stores user behavior events for online learning.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize event buffer.

        Args:
            max_size: Maximum number of events to buffer
        """
        self.max_size = max_size
        self.buffer: queue.Queue = queue.Queue(maxsize=max_size)
        self.lock = threading.Lock()

    def put(self, event: Dict[str, Any]) -> bool:
        """
        Add event to buffer.

        Args:
            event: Event dict with user_id, item_id, label, etc.

        Returns:
            True if added, False if buffer full
        """
        try:
            self.buffer.put_nowait(event)
            return True
        except queue.Full:
            logger.warning("Event buffer is full, dropping event")
            return False

    def get_batch(self, batch_size: int = 256) -> List[Dict[str, Any]]:
        """
        Get batch of events.

        Args:
            batch_size: Number of events to get

        Returns:
            List of events
        """
        events = []
        for _ in range(min(batch_size, self.buffer.qsize())):
            try:
                events.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        return events

    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer.qsize()

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.buffer.full()

    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except queue.Empty:
                    break


class StreamProcessor:
    """
    Stream processor for real-time event handling.
    Processes user behavior events and updates features/models.
    """

    def __init__(self, config=None):
        """
        Initialize stream processor.

        Args:
            config: AppConfig
        """
        self.config = config
        self.buffer = EventBuffer(max_size=10000)
        self.running = False
        self.processors: List[Callable] = []

        # Statistics
        self.events_processed = 0
        self.last_processed_time = None

    def add_processor(self, processor: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add event processor.

        Args:
            processor: Function that processes an event
        """
        self.processors.append(processor)

    def process_event(self, event: Dict[str, Any]) -> None:
        """
        Process a single event.

        Args:
            event: Event dict
        """
        # Apply all processors
        for processor in self.processors:
            try:
                processor(event)
            except Exception as e:
                logger.error(f"Error in event processor: {e}")

        self.events_processed += 1
        self.last_processed_time = datetime.now()

    def push_event(self, event: Dict[str, Any]) -> bool:
        """
        Push event to stream.

        Args:
            event: Event dict

        Returns:
            True if successful
        """
        return self.buffer.put(event)

    def start(self) -> None:
        """Start processing stream."""
        self.running = True
        logger.info("Stream processor started")

        while self.running:
            try:
                events = self.buffer.get_batch(batch_size=100)
                for event in events:
                    self.process_event(event)
                time.sleep(0.1)  # Small delay to prevent busy loop
            except Exception as e:
                logger.error(f"Error in stream loop: {e}")

    def stop(self) -> None:
        """Stop processing stream."""
        self.running = False
        logger.info("Stream processor stopped")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.

        Returns:
            Dict with stats
        """
        return {
            "running": self.running,
            "buffer_size": self.buffer.size(),
            "events_processed": self.events_processed,
            "last_processed_time": self.last_processed_time.isoformat() if self.last_processed_time else None,
        }


class OnlineLearner:
    """
    Online learning module for incremental model updates.
    """

    def __init__(
        self,
        config=None,
        two_tower_model: 'TwoTowerModel' = None,
        ranking_model: 'RankingModel' = None
    ):
        """
        Initialize online learner.

        Args:
            config: AppConfig
            two_tower_model: Two-Tower model for updates
            ranking_model: Ranking model for updates
        """
        self.config = config or get_config()
        self.two_tower = two_tower_model
        self.ranking_model = ranking_model

        # Initialize stream processor
        self.stream = StreamProcessor(self.config)

        # Feature stores
        from features.base import create_feature_store
        self.feature_store = create_feature_store(self.config)

        # Register event processors
        self._setup_processors()

        logger.info("OnlineLearner initialized")

    def _setup_processors(self) -> None:
        """Setup event processors for different event types."""
        self.stream.add_processor(self._process_impression)
        self.stream.add_processor(self._process_click)
        self.stream.add_processor(self._process_rating)

    def _process_impression(self, event: Dict[str, Any]) -> None:
        """Process impression event."""
        user_id = event.get("user_id")
        item_id = event.get("item_id")
        timestamp = event.get("timestamp", datetime.now().isoformat())

        if not user_id or not item_id:
            return

        # Update user's viewed items
        try:
            from features.user_features import UserFeatureBuilder
            builder = UserFeatureBuilder(user_id)

            # Get existing and add new item
            from features.base import UserFeatureStore
            user_store = UserFeatureStore(self.feature_store)
            existing = user_store.get_interaction_history(user_id)

            updated_items = existing + [item_id]
            # Keep only last 1000 items
            updated_items = updated_items[-1000:]

            builder.set_behavior(viewed_items=updated_items)
            self.feature_store.set_user_features(user_id, builder.build())

        except Exception as e:
            logger.error(f"Error processing impression: {e}")

    def _process_click(self, event: Dict[str, Any]) -> None:
        """Process click event."""
        user_id = event.get("user_id")
        item_id = event.get("item_id")

        if not user_id or not item_id:
            return

        # Update item CTR
        try:
            from features.item_features import ItemFeatureBuilder
            builder = ItemFeatureBuilder(item_id)

            # Get existing stats
            item_store = self.feature_store.get_item_features(item_id)
            stats = item_store.get("statistics", {}) if item_store else {}

            views = stats.get("views", 0) + 1
            clicks = stats.get("clicks", 0) + 1
            ctr = clicks / views if views > 0 else 0.05

            builder.set_statistics(
                views=views,
                clicks=clicks,
                ctr=ctr
            )
            self.feature_store.set_item_features(item_id, builder.build())

        except Exception as e:
            logger.error(f"Error processing click: {e}")

    def _process_rating(self, event: Dict[str, Any]) -> None:
        """Process rating event."""
        user_id = event.get("user_id")
        item_id = event.get("item_id")
        rating = event.get("rating")

        if not user_id or not item_id or rating is None:
            return

        # Update user's rating history
        try:
            from features.user_features import UserFeatureBuilder
            builder = UserFeatureBuilder(user_id)
            builder.set_behavior(total_ratings=1)  # Simplified
            self.feature_store.set_user_features(user_id, builder.build())

        except Exception as e:
            logger.error(f"Error processing rating: {e}")

    def push_impression(self, user_id: int, item_id: int) -> bool:
        """
        Push impression event.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            True if successful
        """
        event = {
            "event_type": "impression",
            "user_id": user_id,
            "item_id": item_id,
            "timestamp": datetime.now().isoformat()
        }
        return self.stream.push_event(event)

    def push_click(self, user_id: int, item_id: int) -> bool:
        """
        Push click event.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            True if successful
        """
        event = {
            "event_type": "click",
            "user_id": user_id,
            "item_id": item_id,
            "timestamp": datetime.now().isoformat()
        }
        return self.stream.push_event(event)

    def push_rating(self, user_id: int, item_id: int, rating: float) -> bool:
        """
        Push rating event.

        Args:
            user_id: User ID
            item_id: Item ID
            rating: Rating value

        Returns:
            True if successful
        """
        event = {
            "event_type": "rating",
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        return self.stream.push_event(event)

    def start(self) -> None:
        """Start online learning."""
        self.stream.start()

    def stop(self) -> None:
        """Stop online learning."""
        self.stream.stop()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get learner statistics.

        Returns:
            Dict with stats
        """
        return {
            "stream": self.stream.get_stats(),
            "two_tower_available": self.two_tower is not None,
            "ranking_available": self.ranking_model is not None,
        }

    def update_user_embedding(self, user_id: int, embedding: List[float]) -> bool:
        """
        Update user embedding in cache.

        Args:
            user_id: User ID
            embedding: New embedding

        Returns:
            True if successful
        """
        return self.feature_store.set_user_embedding(user_id, embedding)

    def update_item_embedding(self, item_id: int, embedding: List[float]) -> bool:
        """
        Update item embedding in cache and FAISS.

        Args:
            item_id: Item ID
            embedding: New embedding

        Returns:
            True if successful
        """
        result = self.feature_store.set_item_embedding(item_id, embedding)

        # Also update FAISS if available
        try:
            from serving.faiss_client import get_faiss_client
            faiss = get_faiss_client(self.config.recall.faiss)
            if faiss.is_available():
                faiss.update_item(item_id, embedding)
        except Exception as e:
            logger.warning(f"Failed to update FAISS: {e}")

        return result

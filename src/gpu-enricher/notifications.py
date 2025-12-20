"""
Notification Module for AI FinOps Platform

Provides notification delivery with retry logic for:
- Webhook notifications (generic HTTP endpoints)
- Slack notifications
- PagerDuty alerts

Implements exponential backoff and dead letter queue for failed notifications.
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from threading import Thread
from queue import Queue, Empty

import requests

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications."""

    BUDGET_WARNING = "budget_warning"
    BUDGET_CRITICAL = "budget_critical"
    BUDGET_EXCEEDED = "budget_exceeded"
    ANOMALY_DETECTED = "anomaly_detected"
    IDLE_GPU = "idle_gpu"
    COST_SPIKE = "cost_spike"


class NotificationChannel(Enum):
    """Notification delivery channels."""

    WEBHOOK = "webhook"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel."""

    channel: NotificationChannel
    url: str
    enabled: bool = True
    headers: dict = field(default_factory=dict)
    retry_count: int = 3
    retry_delay_base: float = 1.0  # Base delay in seconds
    retry_delay_max: float = 60.0  # Max delay in seconds
    timeout: float = 10.0


@dataclass
class Notification:
    """A notification to be sent."""

    id: str
    type: NotificationType
    title: str
    message: str
    severity: str  # info, warning, critical
    team: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempt: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "team": self.team,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def to_slack_payload(self) -> dict:
        """Convert to Slack message format."""
        color = {
            "info": "#36a64f",
            "warning": "#ffcc00",
            "critical": "#ff0000",
        }.get(self.severity, "#808080")

        return {
            "attachments": [
                {
                    "color": color,
                    "title": self.title,
                    "text": self.message,
                    "fields": [
                        {"title": "Type", "value": self.type.value, "short": True},
                        {"title": "Severity", "value": self.severity, "short": True},
                        {"title": "Team", "value": self.team or "N/A", "short": True},
                    ],
                    "footer": "AI FinOps Platform",
                    "ts": int(self.created_at.timestamp()),
                }
            ]
        }

    def to_pagerduty_payload(self, routing_key: str) -> dict:
        """Convert to PagerDuty Events API v2 format."""
        severity_map = {
            "info": "info",
            "warning": "warning",
            "critical": "critical",
        }

        return {
            "routing_key": routing_key,
            "event_action": "trigger",
            "dedup_key": self.id,
            "payload": {
                "summary": f"{self.title}: {self.message}",
                "severity": severity_map.get(self.severity, "warning"),
                "source": "ai-finops-platform",
                "component": "gpu-enricher",
                "group": self.team or "default",
                "class": self.type.value,
                "custom_details": self.metadata,
            },
        }


@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue for failed notifications."""

    notification: Notification
    channel: NotificationChannel
    error: str
    failed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: int = 0


class NotificationSender:
    """
    Sends notifications with retry logic.

    Features:
    - Exponential backoff for retries
    - Dead letter queue for failed notifications
    - Async sending via background thread
    - Rate limiting per channel
    """

    def __init__(self):
        self._configs: dict[NotificationChannel, NotificationConfig] = {}
        self._queue: Queue = Queue()
        self._dead_letter_queue: list[DeadLetterEntry] = []
        self._running = False
        self._worker_thread: Optional[Thread] = None
        self._rate_limits: dict[str, list[float]] = {}
        self._rate_limit_window = 60  # seconds
        self._rate_limit_max = 30  # max notifications per window

    def configure_webhook(
        self,
        url: str,
        headers: Optional[dict] = None,
        enabled: bool = True,
    ) -> None:
        """Configure generic webhook notification."""
        self._configs[NotificationChannel.WEBHOOK] = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            url=url,
            enabled=enabled,
            headers=headers or {"Content-Type": "application/json"},
        )
        logger.info(f"Webhook notifications configured: {url}")

    def configure_slack(
        self,
        webhook_url: str,
        enabled: bool = True,
    ) -> None:
        """Configure Slack webhook notification."""
        self._configs[NotificationChannel.SLACK] = NotificationConfig(
            channel=NotificationChannel.SLACK,
            url=webhook_url,
            enabled=enabled,
            headers={"Content-Type": "application/json"},
        )
        logger.info("Slack notifications configured")

    def configure_pagerduty(
        self,
        routing_key: str,
        enabled: bool = True,
    ) -> None:
        """Configure PagerDuty notification."""
        self._configs[NotificationChannel.PAGERDUTY] = NotificationConfig(
            channel=NotificationChannel.PAGERDUTY,
            url="https://events.pagerduty.com/v2/enqueue",
            enabled=enabled,
            headers={
                "Content-Type": "application/json",
                "X-Routing-Key": routing_key,
            },
        )
        logger.info("PagerDuty notifications configured")

    def start(self) -> None:
        """Start the background notification worker."""
        if self._running:
            return

        self._running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Notification sender started")

    def stop(self) -> None:
        """Stop the background notification worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Notification sender stopped")

    def send(
        self,
        notification: Notification,
        channels: Optional[list[NotificationChannel]] = None,
    ) -> None:
        """
        Queue a notification for sending.

        Args:
            notification: The notification to send
            channels: List of channels to send to (default: all configured)
        """
        if channels is None:
            channels = list(self._configs.keys())

        for channel in channels:
            if channel in self._configs and self._configs[channel].enabled:
                self._queue.put((notification, channel))

    def send_sync(
        self,
        notification: Notification,
        channel: NotificationChannel,
    ) -> bool:
        """
        Send a notification synchronously with retries.

        Args:
            notification: The notification to send
            channel: The channel to send to

        Returns:
            True if sent successfully, False otherwise
        """
        config = self._configs.get(channel)
        if not config or not config.enabled:
            logger.warning(f"Channel {channel} not configured or disabled")
            return False

        # Check rate limit
        if not self._check_rate_limit(channel.value):
            logger.warning(f"Rate limit exceeded for channel {channel}")
            return False

        for attempt in range(config.retry_count):
            try:
                self._send_notification(notification, config)
                logger.info(
                    f"Notification {notification.id} sent via {channel.value} "
                    f"(attempt {attempt + 1})"
                )
                return True
            except Exception as e:
                delay = self._calculate_backoff(attempt, config)
                logger.warning(
                    f"Failed to send notification {notification.id} via {channel.value} "
                    f"(attempt {attempt + 1}/{config.retry_count}): {e}"
                )
                notification.last_error = str(e)
                notification.attempt = attempt + 1

                if attempt < config.retry_count - 1:
                    time.sleep(delay)

        # Add to dead letter queue
        self._dead_letter_queue.append(
            DeadLetterEntry(
                notification=notification,
                channel=channel,
                error=notification.last_error or "Unknown error",
                attempts=config.retry_count,
            )
        )
        logger.error(
            f"Notification {notification.id} added to dead letter queue "
            f"after {config.retry_count} attempts"
        )
        return False

    def _worker_loop(self) -> None:
        """Background worker loop for processing notifications."""
        while self._running:
            try:
                notification, channel = self._queue.get(timeout=1)
                self.send_sync(notification, channel)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")

    def _send_notification(
        self,
        notification: Notification,
        config: NotificationConfig,
    ) -> None:
        """Send a single notification attempt."""
        if config.channel == NotificationChannel.SLACK:
            payload = notification.to_slack_payload()
        elif config.channel == NotificationChannel.PAGERDUTY:
            routing_key = config.headers.get("X-Routing-Key", "")
            payload = notification.to_pagerduty_payload(routing_key)
        else:
            payload = notification.to_dict()

        response = requests.post(
            config.url,
            json=payload,
            headers=config.headers,
            timeout=config.timeout,
        )
        response.raise_for_status()

    def _calculate_backoff(self, attempt: int, config: NotificationConfig) -> float:
        """Calculate exponential backoff delay."""
        delay = config.retry_delay_base * (2**attempt)
        return min(delay, config.retry_delay_max)

    def _check_rate_limit(self, channel_key: str) -> bool:
        """Check if we're within rate limits for a channel."""
        now = time.time()

        if channel_key not in self._rate_limits:
            self._rate_limits[channel_key] = []

        # Remove old entries
        self._rate_limits[channel_key] = [
            t
            for t in self._rate_limits[channel_key]
            if now - t < self._rate_limit_window
        ]

        if len(self._rate_limits[channel_key]) >= self._rate_limit_max:
            return False

        self._rate_limits[channel_key].append(now)
        return True

    def get_dead_letter_queue(self) -> list[dict]:
        """Get the dead letter queue contents."""
        return [
            {
                "notification_id": entry.notification.id,
                "channel": entry.channel.value,
                "error": entry.error,
                "failed_at": entry.failed_at.isoformat(),
                "attempts": entry.attempts,
            }
            for entry in self._dead_letter_queue
        ]

    def retry_dead_letters(self) -> dict:
        """Retry all notifications in the dead letter queue."""
        results = {"retried": 0, "succeeded": 0, "failed": 0}

        entries = self._dead_letter_queue.copy()
        self._dead_letter_queue.clear()

        for entry in entries:
            results["retried"] += 1
            if self.send_sync(entry.notification, entry.channel):
                results["succeeded"] += 1
            else:
                results["failed"] += 1

        return results

    def get_stats(self) -> dict:
        """Get notification sender statistics."""
        return {
            "configured_channels": [c.value for c in self._configs.keys()],
            "enabled_channels": [
                c.value for c, cfg in self._configs.items() if cfg.enabled
            ],
            "queue_size": self._queue.qsize(),
            "dead_letter_queue_size": len(self._dead_letter_queue),
            "running": self._running,
        }


# Global notification sender instance
notification_sender = NotificationSender()


def initialize_notifications() -> None:
    """Initialize notification sender from environment variables."""
    # Generic webhook
    webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL")
    if webhook_url:
        notification_sender.configure_webhook(url=webhook_url)

    # Slack
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        notification_sender.configure_slack(webhook_url=slack_webhook)

    # PagerDuty
    pagerduty_key = os.getenv("PAGERDUTY_ROUTING_KEY")
    if pagerduty_key:
        notification_sender.configure_pagerduty(routing_key=pagerduty_key)

    # Start the sender
    notification_sender.start()
    logger.info(f"Notifications initialized: {notification_sender.get_stats()}")


def create_notification(
    type: NotificationType,
    title: str,
    message: str,
    severity: str = "info",
    team: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Notification:
    """Create a new notification with a unique ID."""
    # Generate deterministic ID based on content
    content = f"{type.value}:{title}:{team}:{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')}"
    notification_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    return Notification(
        id=notification_id,
        type=type,
        title=title,
        message=message,
        severity=severity,
        team=team,
        metadata=metadata or {},
    )

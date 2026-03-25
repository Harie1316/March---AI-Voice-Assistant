"""
Third-Party Integration Service.

Provides a pluggable framework for integrating with external services
via RESTful API calls. Supports webhook registration, authentication
management, and response normalisation.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for a third-party integration."""
    id: str
    name: str
    base_url: str
    auth_type: str = "bearer"   # bearer, api_key, basic, none
    auth_token: str = ""
    headers: dict = field(default_factory=dict)
    enabled: bool = True
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "auth_type": self.auth_type,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }


@dataclass
class IntegrationResult:
    """Result of an integration call."""
    success: bool
    integration_id: str
    data: Any = None
    error: Optional[str] = None
    status_code: int = 200
    response_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "integration_id": self.integration_id,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
        }


@dataclass
class Webhook:
    """Webhook registration for event notifications."""
    id: str
    url: str
    events: list[str]
    active: bool = True
    secret: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.secret:
            self.secret = uuid.uuid4().hex[:32]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "events": self.events,
            "active": self.active,
            "created_at": self.created_at,
        }


class IntegrationService:
    """
    Manages third-party integrations, webhook delivery, and API routing.
    """

    # Built-in integration templates
    TEMPLATES = {
        "slack": {
            "name": "Slack",
            "base_url": "https://slack.com/api/",
            "auth_type": "bearer",
            "endpoints": {
                "send_message": "chat.postMessage",
                "list_channels": "conversations.list",
            },
        },
        "telegram": {
            "name": "Telegram Bot",
            "base_url": "https://api.telegram.org/bot{token}/",
            "auth_type": "none",
            "endpoints": {
                "send_message": "sendMessage",
                "get_updates": "getUpdates",
            },
        },
        "ifttt": {
            "name": "IFTTT",
            "base_url": "https://maker.ifttt.com/trigger/",
            "auth_type": "api_key",
            "endpoints": {
                "trigger": "{event}/with/key/{key}",
            },
        },
        "home_assistant": {
            "name": "Home Assistant",
            "base_url": "http://homeassistant.local:8123/api/",
            "auth_type": "bearer",
            "endpoints": {
                "call_service": "services/{domain}/{service}",
                "get_states": "states",
            },
        },
    }

    def __init__(self):
        self._integrations: dict[str, IntegrationConfig] = {}
        self._webhooks: dict[str, Webhook] = {}
        logger.info("IntegrationService initialised")

    def register_integration(
        self,
        name: str,
        base_url: str,
        auth_type: str = "bearer",
        auth_token: str = "",
        headers: Optional[dict] = None,
    ) -> IntegrationConfig:
        """Register a new third-party integration."""
        integration_id = str(uuid.uuid4())[:12]

        config = IntegrationConfig(
            id=integration_id,
            name=name,
            base_url=base_url,
            auth_type=auth_type,
            auth_token=auth_token,
            headers=headers or {},
        )

        self._integrations[integration_id] = config
        logger.info("Integration registered: %s (%s)", name, integration_id)
        return config

    def register_from_template(
        self, template_name: str, auth_token: str = ""
    ) -> Optional[IntegrationConfig]:
        """Register an integration from a built-in template."""
        template = self.TEMPLATES.get(template_name)
        if not template:
            logger.warning("Unknown integration template: %s", template_name)
            return None

        return self.register_integration(
            name=template["name"],
            base_url=template["base_url"],
            auth_type=template["auth_type"],
            auth_token=auth_token,
        )

    def call_integration(
        self,
        integration_id: str,
        endpoint: str,
        method: str = "POST",
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> IntegrationResult:
        """
        Make an API call to a registered integration.
        
        In production, this would use httpx/requests. Here we provide
        the framework with proper auth header construction and error handling.
        """
        start_time = time.perf_counter()

        config = self._integrations.get(integration_id)
        if not config:
            return IntegrationResult(
                success=False,
                integration_id=integration_id,
                error="Integration not found",
                status_code=404,
            )

        if not config.enabled:
            return IntegrationResult(
                success=False,
                integration_id=integration_id,
                error="Integration is disabled",
                status_code=403,
            )

        # Build request
        url = urljoin(config.base_url, endpoint)
        headers = dict(config.headers)
        headers["Content-Type"] = "application/json"

        if config.auth_type == "bearer" and config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"
        elif config.auth_type == "api_key" and config.auth_token:
            headers["X-API-Key"] = config.auth_token

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # In production: make actual HTTP request
        # For now, return a simulated success
        logger.info(
            "Integration call: %s %s (%s) — %.0fms",
            method, url, config.name, elapsed_ms,
        )

        return IntegrationResult(
            success=True,
            integration_id=integration_id,
            data={
                "message": f"Request prepared for {config.name}",
                "url": url,
                "method": method,
                "payload": data,
            },
            status_code=200,
            response_time_ms=elapsed_ms,
        )

    def register_webhook(
        self, url: str, events: list[str]
    ) -> Webhook:
        """Register a webhook for event notifications."""
        webhook_id = str(uuid.uuid4())[:12]
        webhook = Webhook(id=webhook_id, url=url, events=events)
        self._webhooks[webhook_id] = webhook
        logger.info("Webhook registered: %s for events %s", webhook_id, events)
        return webhook

    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        return self._webhooks.get(webhook_id)

    def list_webhooks(self) -> list[Webhook]:
        return list(self._webhooks.values())

    def list_integrations(self) -> list[IntegrationConfig]:
        return list(self._integrations.values())

    def get_integration(self, integration_id: str) -> Optional[IntegrationConfig]:
        return self._integrations.get(integration_id)

    def disable_integration(self, integration_id: str) -> bool:
        config = self._integrations.get(integration_id)
        if config:
            config.enabled = False
            return True
        return False

    def get_available_templates(self) -> list[dict]:
        return [
            {"key": k, "name": v["name"], "base_url": v["base_url"]}
            for k, v in self.TEMPLATES.items()
        ]

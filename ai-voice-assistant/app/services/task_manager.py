"""
Task Automation Service.

Manages task creation, scheduling, and execution for the voice assistant.
Supports task queuing, status tracking, and webhook-based notifications.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    REMINDER = "reminder"
    TIMER = "timer"
    EMAIL = "email"
    CALENDAR = "calendar"
    SMART_HOME = "smart_home"
    SEARCH = "search"
    CUSTOM = "custom"


@dataclass
class Task:
    """Represents an automation task."""
    id: str
    type: TaskType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = ""
    updated_at: str = ""
    scheduled_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    webhook_url: Optional[str] = None

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "scheduled_at": self.scheduled_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskManager:
    """
    Manages task lifecycle: creation, execution, tracking, and completion.
    Thread-safe for concurrent request handling.
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._lock = Lock()
        logger.info("TaskManager initialised")

    def create_task(
        self,
        task_type: str,
        description: str,
        scheduled_at: Optional[str] = None,
        metadata: Optional[dict] = None,
        webhook_url: Optional[str] = None,
    ) -> Task:
        """Create a new automation task."""
        task_id = str(uuid.uuid4())[:12]

        try:
            ttype = TaskType(task_type)
        except ValueError:
            ttype = TaskType.CUSTOM

        task = Task(
            id=task_id,
            type=ttype,
            description=description,
            scheduled_at=scheduled_at,
            metadata=metadata or {},
            webhook_url=webhook_url,
        )

        with self._lock:
            self._tasks[task_id] = task

        logger.info("Task created: %s (%s) — %s", task_id, ttype.value, description)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> Optional[Task]:
        """Update task status."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None

            task.status = status
            task.updated_at = datetime.utcnow().isoformat()

            if status == TaskStatus.COMPLETED:
                task.completed_at = task.updated_at
                task.result = result
            elif status == TaskStatus.FAILED:
                task.error = error

        logger.info("Task %s updated to %s", task_id, status.value)
        return task

    def list_tasks(
        self,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[Task]:
        """List tasks with optional filtering."""
        with self._lock:
            tasks = list(self._tasks.values())

        if status:
            tasks = [t for t in tasks if t.status.value == status]
        if task_type:
            tasks = [t for t in tasks if t.type.value == task_type]

        # Sort by creation time descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def cancel_task(self, task_id: str) -> Optional[Task]:
        """Cancel a pending task."""
        return self.update_task_status(task_id, TaskStatus.CANCELLED)

    def execute_task(self, task_id: str) -> Optional[Task]:
        """
        Execute a task (simulated — in production, dispatches to real services).
        """
        task = self.get_task(task_id)
        if not task:
            return None

        if task.status != TaskStatus.PENDING:
            logger.warning("Task %s is not pending (status=%s)", task_id, task.status.value)
            return task

        self.update_task_status(task_id, TaskStatus.RUNNING)

        try:
            # Simulate task execution by type
            result = self._dispatch_task(task)
            self.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
        except Exception as e:
            logger.error("Task %s failed: %s", task_id, e)
            self.update_task_status(task_id, TaskStatus.FAILED, error=str(e))

        return self.get_task(task_id)

    def _dispatch_task(self, task: Task) -> dict:
        """Dispatch task to appropriate handler."""
        handlers = {
            TaskType.REMINDER: self._handle_reminder,
            TaskType.TIMER: self._handle_timer,
            TaskType.EMAIL: self._handle_email,
            TaskType.CALENDAR: self._handle_calendar,
            TaskType.SMART_HOME: self._handle_smart_home,
            TaskType.SEARCH: self._handle_search,
            TaskType.CUSTOM: self._handle_custom,
        }

        handler = handlers.get(task.type, self._handle_custom)
        return handler(task)

    def _handle_reminder(self, task: Task) -> dict:
        return {
            "action": "reminder_set",
            "message": task.description,
            "scheduled_at": task.scheduled_at or "now",
        }

    def _handle_timer(self, task: Task) -> dict:
        duration = task.metadata.get("duration_seconds", 60)
        return {"action": "timer_set", "duration_seconds": duration}

    def _handle_email(self, task: Task) -> dict:
        return {
            "action": "email_queued",
            "to": task.metadata.get("to", ""),
            "subject": task.metadata.get("subject", task.description),
        }

    def _handle_calendar(self, task: Task) -> dict:
        return {
            "action": "event_created",
            "title": task.description,
            "datetime": task.scheduled_at,
        }

    def _handle_smart_home(self, task: Task) -> dict:
        return {
            "action": "device_command_sent",
            "device": task.metadata.get("device", ""),
            "command": task.metadata.get("command", ""),
        }

    def _handle_search(self, task: Task) -> dict:
        return {
            "action": "search_completed",
            "query": task.description,
            "results_count": 10,
        }

    def _handle_custom(self, task: Task) -> dict:
        return {"action": "custom_completed", "description": task.description}

    def get_stats(self) -> dict:
        """Get task statistics."""
        with self._lock:
            tasks = list(self._tasks.values())

        return {
            "total": len(tasks),
            "by_status": {
                s.value: len([t for t in tasks if t.status == s])
                for s in TaskStatus
            },
            "by_type": {
                tt.value: len([t for t in tasks if t.type == tt])
                for tt in TaskType
            },
        }

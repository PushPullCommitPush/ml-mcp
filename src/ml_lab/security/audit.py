"""Audit logging for tracking all sensitive operations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AuditCategory(str, Enum):
    """Categories of audit events."""

    CREDENTIAL = "credential"  # Vault operations
    VPS = "vps"  # Remote VPS operations
    TRAINING = "training"  # Training launches
    CLOUD = "cloud"  # Cloud provider operations
    INFERENCE = "inference"  # Ollama/OpenWebUI operations
    SECURITY = "security"  # Security-related events


class AuditAction(str, Enum):
    """Specific actions that are audited."""

    # Credential actions
    VAULT_CREATE = "vault_create"
    VAULT_UNLOCK = "vault_unlock"
    VAULT_LOCK = "vault_lock"
    CRED_ADD = "cred_add"
    CRED_GET = "cred_get"
    CRED_DELETE = "cred_delete"
    CRED_LIST = "cred_list"

    # VPS actions
    VPS_REGISTER = "vps_register"
    VPS_UNREGISTER = "vps_unregister"
    VPS_CONNECT = "vps_connect"
    VPS_RUN_COMMAND = "vps_run_command"
    VPS_SYNC = "vps_sync"
    VPS_KEY_ROTATE = "vps_key_rotate"

    # Training actions
    TRAIN_LAUNCH = "train_launch"
    TRAIN_STOP = "train_stop"
    TRAIN_STATUS = "train_status"

    # Cloud actions
    CLOUD_PROVISION = "cloud_provision"
    CLOUD_TERMINATE = "cloud_terminate"
    CLOUD_QUERY = "cloud_query"

    # Inference actions
    OLLAMA_DEPLOY = "ollama_deploy"
    OLLAMA_DELETE = "ollama_delete"
    OWUI_CREATE = "owui_create"
    OWUI_DELETE = "owui_delete"

    # Security actions
    TAILSCALE_CHECK = "tailscale_check"
    SSH_KEY_ROTATE = "ssh_key_rotate"


@dataclass
class AuditEvent:
    """A single audit log entry."""

    timestamp: str
    category: str
    action: str
    target: str | None  # e.g., VPS name, provider name
    success: bool
    details: dict[str, Any] | None = None
    error: str | None = None
    user: str | None = None  # System user who triggered the action

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, json_str: str) -> "AuditEvent":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class AuditLog:
    """
    Append-only audit log for tracking sensitive operations.

    Logs are stored in ~/.cache/ml-lab/audit.log as newline-delimited JSON.
    Each line is a complete, parseable JSON object.
    """

    def __init__(self, log_path: Path | None = None):
        """Initialize the audit log."""
        if log_path is None:
            log_path = Path.home() / ".cache" / "ml-lab" / "audit.log"
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Get current system user
        self._user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"

    def log(
        self,
        category: AuditCategory,
        action: AuditAction,
        target: str | None = None,
        success: bool = True,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            category: Event category.
            action: Specific action.
            target: Target of the action (e.g., VPS name).
            success: Whether the action succeeded.
            details: Additional details (sanitized - no secrets).
            error: Error message if failed.

        Returns:
            The created AuditEvent.
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            category=category.value,
            action=action.value,
            target=target,
            success=success,
            details=details,
            error=error,
            user=self._user,
        )

        # Append to log file
        with open(self._log_path, "a") as f:
            f.write(event.to_json() + "\n")

        return event

    def get_recent(
        self,
        limit: int = 100,
        category: AuditCategory | None = None,
        action: AuditAction | None = None,
        target: str | None = None,
        success_only: bool = False,
        failures_only: bool = False,
    ) -> list[AuditEvent]:
        """
        Get recent audit events with optional filtering.

        Args:
            limit: Maximum number of events to return.
            category: Filter by category.
            action: Filter by action.
            target: Filter by target.
            success_only: Only return successful events.
            failures_only: Only return failed events.

        Returns:
            List of matching AuditEvent objects (newest first).
        """
        if not self._log_path.exists():
            return []

        events = []

        # Read file in reverse (newest first)
        with open(self._log_path, "rb") as f:
            # Seek to end
            f.seek(0, 2)
            file_size = f.tell()

            if file_size == 0:
                return []

            # Read backwards line by line
            buffer = b""
            position = file_size

            while position > 0 and len(events) < limit:
                # Read in chunks
                chunk_size = min(4096, position)
                position -= chunk_size
                f.seek(position)
                chunk = f.read(chunk_size)
                buffer = chunk + buffer

                # Process complete lines
                while b"\n" in buffer and len(events) < limit:
                    line, buffer = buffer.rsplit(b"\n", 1)
                    if line:
                        try:
                            event = AuditEvent.from_json(line.decode())

                            # Apply filters
                            if category and event.category != category.value:
                                continue
                            if action and event.action != action.value:
                                continue
                            if target and event.target != target:
                                continue
                            if success_only and not event.success:
                                continue
                            if failures_only and event.success:
                                continue

                            events.append(event)
                        except (json.JSONDecodeError, TypeError):
                            continue

            # Process remaining buffer
            if buffer and len(events) < limit:
                try:
                    event = AuditEvent.from_json(buffer.decode())
                    if not (category and event.category != category.value):
                        if not (action and event.action != action.value):
                            if not (target and event.target != target):
                                if not (success_only and not event.success):
                                    if not (failures_only and event.success):
                                        events.append(event)
                except (json.JSONDecodeError, TypeError):
                    pass

        return events

    def get_summary(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get a summary of audit activity.

        Args:
            since: Only include events after this time.

        Returns:
            Summary dict with counts by category and action.
        """
        if not self._log_path.exists():
            return {"total": 0, "by_category": {}, "by_action": {}, "failures": 0}

        total = 0
        failures = 0
        by_category: dict[str, int] = {}
        by_action: dict[str, int] = {}

        with open(self._log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = AuditEvent.from_json(line)

                    # Filter by time if specified
                    if since:
                        event_time = datetime.fromisoformat(
                            event.timestamp.rstrip("Z")
                        )
                        if event_time < since:
                            continue

                    total += 1
                    if not event.success:
                        failures += 1

                    by_category[event.category] = by_category.get(event.category, 0) + 1
                    by_action[event.action] = by_action.get(event.action, 0) + 1

                except (json.JSONDecodeError, TypeError):
                    continue

        return {
            "total": total,
            "failures": failures,
            "by_category": by_category,
            "by_action": by_action,
        }

    def clear(self, before: datetime | None = None) -> int:
        """
        Clear audit log entries.

        Args:
            before: Only clear entries before this time. If None, clears all.

        Returns:
            Number of entries cleared.
        """
        if not self._log_path.exists():
            return 0

        if before is None:
            # Clear entire log
            with open(self._log_path) as f:
                count = sum(1 for _ in f)
            self._log_path.unlink()
            return count

        # Keep entries after the cutoff
        kept_lines = []
        cleared = 0

        with open(self._log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = AuditEvent.from_json(line)
                    event_time = datetime.fromisoformat(event.timestamp.rstrip("Z"))
                    if event_time >= before:
                        kept_lines.append(line)
                    else:
                        cleared += 1
                except (json.JSONDecodeError, TypeError, ValueError):
                    kept_lines.append(line)  # Keep unparseable lines

        with open(self._log_path, "w") as f:
            f.writelines(kept_lines)

        return cleared


# Singleton instance
_audit_log: AuditLog | None = None


def get_audit_log() -> AuditLog:
    """Get or create the global audit log."""
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog()
    return _audit_log

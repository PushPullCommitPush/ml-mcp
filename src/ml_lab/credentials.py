"""
Secure credential vault for API keys and secrets.

Credentials are encrypted at rest using Fernet (AES-128-CBC).
The encryption key is derived from a user-provided password or
stored in the system keyring.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# Default expiry periods for different provider types (in days)
DEFAULT_EXPIRY_DAYS: dict[str, int] = {
    "lambda_labs": 365,
    "runpod": 365,
    "vast_ai": 365,
    "modal": 365,
    "gcp": 90,
    "aws": 90,
    "mistral": 365,
    "openai": 365,
    "together": 365,
    "fireworks": 365,
    "anyscale": 365,
    "huggingface": 365,
    "replicate": 365,
    "ollama": None,  # Local, no expiry
    "remote_vps": 180,  # SSH keys should be rotated
}


class ProviderType(str, Enum):
    """Types of credential providers."""

    # Compute providers (raw GPU access)
    LAMBDA_LABS = "lambda_labs"
    RUNPOD = "runpod"
    VAST_AI = "vast_ai"
    MODAL = "modal"
    GCP = "gcp"
    AWS = "aws"

    # Fine-tuning APIs
    MISTRAL = "mistral"
    OPENAI = "openai"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    ANYSCALE = "anyscale"

    # Model hubs
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"

    # Local
    OLLAMA = "ollama"

    # Remote VPS (generic SSH-accessible machines)
    REMOTE_VPS = "remote_vps"


@dataclass
class ProviderCredential:
    """A single provider's credentials."""

    provider: ProviderType
    api_key: str | None = None
    api_secret: str | None = None
    ssh_key_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Expiry tracking
    created_at: str | None = None  # ISO format timestamp
    expires_at: str | None = None  # ISO format timestamp
    last_rotated: str | None = None  # ISO format timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "ssh_key_path": self.ssh_key_path,
            "extra": self.extra,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_rotated": self.last_rotated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderCredential:
        return cls(
            provider=ProviderType(data["provider"]),
            api_key=data.get("api_key"),
            api_secret=data.get("api_secret"),
            ssh_key_path=data.get("ssh_key_path"),
            extra=data.get("extra", {}),
            created_at=data.get("created_at"),
            expires_at=data.get("expires_at"),
            last_rotated=data.get("last_rotated"),
        )

    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if not self.expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(self.expires_at.rstrip("Z"))
            return datetime.utcnow() > expiry
        except ValueError:
            return False

    def days_until_expiry(self) -> int | None:
        """Get days until expiry, or None if no expiry set."""
        if not self.expires_at:
            return None
        try:
            expiry = datetime.fromisoformat(self.expires_at.rstrip("Z"))
            delta = expiry - datetime.utcnow()
            return max(0, delta.days)
        except ValueError:
            return None

    def is_expiring_soon(self, days: int = 30) -> bool:
        """Check if credential expires within the given number of days."""
        remaining = self.days_until_expiry()
        if remaining is None:
            return False
        return remaining <= days


class CredentialVault:
    """
    Encrypted credential storage.

    Credentials are stored in an encrypted JSON file. The encryption key
    is derived from a password using PBKDF2.
    """

    SALT_SIZE = 16
    ITERATIONS = 480_000  # OWASP recommendation for PBKDF2-SHA256

    def __init__(self, vault_path: Path | None = None):
        """
        Initialize the credential vault.

        Args:
            vault_path: Path to the encrypted vault file. Defaults to
                       ~/.config/ml-lab/credentials.enc
        """
        if vault_path is None:
            config_dir = Path.home() / ".config" / "ml-lab"
            config_dir.mkdir(parents=True, exist_ok=True)
            vault_path = config_dir / "credentials.enc"

        self.vault_path = vault_path
        self._credentials: dict[ProviderType, ProviderCredential] = {}
        self._fernet: Fernet | None = None
        self._unlocked = False

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive an encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def create(self, password: str) -> None:
        """
        Create a new vault with the given password.

        Args:
            password: The password to encrypt the vault with.
        """
        salt = os.urandom(self.SALT_SIZE)
        key = self._derive_key(password, salt)
        self._fernet = Fernet(key)
        self._credentials = {}
        self._unlocked = True
        self._save(salt)

    def unlock(self, password: str) -> bool:
        """
        Unlock an existing vault.

        Args:
            password: The vault password.

        Returns:
            True if unlock succeeded, False otherwise.
        """
        if not self.vault_path.exists():
            return False

        try:
            with open(self.vault_path, "rb") as f:
                data = f.read()

            salt = data[:self.SALT_SIZE]
            encrypted = data[self.SALT_SIZE:]

            key = self._derive_key(password, salt)
            self._fernet = Fernet(key)

            decrypted = self._fernet.decrypt(encrypted)
            creds_data = json.loads(decrypted.decode())

            self._credentials = {
                ProviderType(k): ProviderCredential.from_dict(v)
                for k, v in creds_data.items()
            }
            self._unlocked = True
            return True

        except Exception:
            self._fernet = None
            self._credentials = {}
            self._unlocked = False
            return False

    def lock(self) -> None:
        """Lock the vault, clearing credentials from memory."""
        self._credentials = {}
        self._fernet = None
        self._unlocked = False

    def _save(self, salt: bytes | None = None) -> None:
        """Save credentials to the encrypted vault file."""
        if not self._unlocked or self._fernet is None:
            raise RuntimeError("Vault is locked")

        creds_data = {k.value: v.to_dict() for k, v in self._credentials.items()}
        plaintext = json.dumps(creds_data).encode()
        encrypted = self._fernet.encrypt(plaintext)

        if salt is None:
            # Read existing salt
            with open(self.vault_path, "rb") as f:
                salt = f.read(self.SALT_SIZE)

        with open(self.vault_path, "wb") as f:
            f.write(salt + encrypted)

        # Secure file permissions (owner read/write only)
        self.vault_path.chmod(0o600)

    @property
    def is_unlocked(self) -> bool:
        """Check if the vault is unlocked."""
        return self._unlocked

    @property
    def exists(self) -> bool:
        """Check if the vault file exists."""
        return self.vault_path.exists()

    def add(
        self,
        credential: ProviderCredential,
        expiry_days: int | None = None,
    ) -> None:
        """
        Add or update a credential.

        Args:
            credential: The credential to store.
            expiry_days: Days until expiry (uses default for provider if not set).
        """
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        now = datetime.utcnow()

        # Set created_at if not already set
        if not credential.created_at:
            credential.created_at = now.isoformat() + "Z"

        # Set expiry if not set
        if not credential.expires_at:
            if expiry_days is None:
                expiry_days = DEFAULT_EXPIRY_DAYS.get(credential.provider.value)
            if expiry_days:
                expiry = now + timedelta(days=expiry_days)
                credential.expires_at = expiry.isoformat() + "Z"

        self._credentials[credential.provider] = credential
        self._save()

    def get(self, provider: ProviderType) -> ProviderCredential | None:
        """
        Get a credential by provider type.

        Args:
            provider: The provider to get credentials for.

        Returns:
            The credential if found, None otherwise.
        """
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        return self._credentials.get(provider)

    def remove(self, provider: ProviderType) -> bool:
        """
        Remove a credential.

        Args:
            provider: The provider to remove.

        Returns:
            True if removed, False if not found.
        """
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        if provider in self._credentials:
            del self._credentials[provider]
            self._save()
            return True
        return False

    def list_providers(self) -> list[ProviderType]:
        """List all stored provider types."""
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        return list(self._credentials.keys())

    def has(self, provider: ProviderType) -> bool:
        """Check if credentials exist for a provider."""
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        return provider in self._credentials

    def get_expiring(self, days: int = 30) -> list[ProviderCredential]:
        """
        Get credentials expiring within the specified number of days.

        Args:
            days: Number of days to check ahead.

        Returns:
            List of credentials expiring soon.
        """
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        return [
            cred for cred in self._credentials.values()
            if cred.is_expiring_soon(days)
        ]

    def get_expired(self) -> list[ProviderCredential]:
        """Get all expired credentials."""
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        return [
            cred for cred in self._credentials.values()
            if cred.is_expired()
        ]

    def rotate_credential(
        self,
        provider: ProviderType,
        new_api_key: str | None = None,
        new_api_secret: str | None = None,
        new_ssh_key_path: str | None = None,
        expiry_days: int | None = None,
    ) -> ProviderCredential | None:
        """
        Rotate a credential, updating last_rotated timestamp.

        Args:
            provider: Provider to rotate.
            new_api_key: New API key.
            new_api_secret: New API secret.
            new_ssh_key_path: New SSH key path.
            expiry_days: Days until new expiry.

        Returns:
            Updated credential, or None if not found.
        """
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        existing = self._credentials.get(provider)
        if not existing:
            return None

        now = datetime.utcnow()

        # Update fields
        if new_api_key:
            existing.api_key = new_api_key
        if new_api_secret:
            existing.api_secret = new_api_secret
        if new_ssh_key_path:
            existing.ssh_key_path = new_ssh_key_path

        # Update rotation timestamp
        existing.last_rotated = now.isoformat() + "Z"

        # Update expiry
        if expiry_days is None:
            expiry_days = DEFAULT_EXPIRY_DAYS.get(provider.value)
        if expiry_days:
            expiry = now + timedelta(days=expiry_days)
            existing.expires_at = expiry.isoformat() + "Z"

        self._save()
        return existing

    def check_expiry_status(self) -> dict[str, Any]:
        """
        Get a summary of credential expiry status.

        Returns:
            Dict with expired, expiring_soon, and healthy counts.
        """
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

        expired = []
        expiring_7d = []
        expiring_30d = []
        healthy = []

        for cred in self._credentials.values():
            if cred.is_expired():
                expired.append(cred.provider.value)
            elif cred.is_expiring_soon(7):
                expiring_7d.append(cred.provider.value)
            elif cred.is_expiring_soon(30):
                expiring_30d.append(cred.provider.value)
            else:
                healthy.append(cred.provider.value)

        return {
            "expired": expired,
            "expiring_7_days": expiring_7d,
            "expiring_30_days": expiring_30d,
            "healthy": healthy,
            "total": len(self._credentials),
        }


# Singleton vault instance
_vault: CredentialVault | None = None


def get_vault(vault_path: Path | None = None) -> CredentialVault:
    """Get or create the global vault instance."""
    global _vault
    if _vault is None:
        _vault = CredentialVault(vault_path)
    return _vault

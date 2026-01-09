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
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "ssh_key_path": self.ssh_key_path,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderCredential:
        return cls(
            provider=ProviderType(data["provider"]),
            api_key=data.get("api_key"),
            api_secret=data.get("api_secret"),
            ssh_key_path=data.get("ssh_key_path"),
            extra=data.get("extra", {}),
        )


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

    def add(self, credential: ProviderCredential) -> None:
        """
        Add or update a credential.

        Args:
            credential: The credential to store.
        """
        if not self._unlocked:
            raise RuntimeError("Vault is locked")

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


# Singleton vault instance
_vault: CredentialVault | None = None


def get_vault(vault_path: Path | None = None) -> CredentialVault:
    """Get or create the global vault instance."""
    global _vault
    if _vault is None:
        _vault = CredentialVault(vault_path)
    return _vault

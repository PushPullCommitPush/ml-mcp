"""Tests for credential vault."""

import tempfile
from pathlib import Path

import pytest

from ml_lab.credentials import CredentialVault, ProviderCredential, ProviderType


class TestCredentialVault:
    def test_create_and_unlock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test.enc"
            vault = CredentialVault(vault_path)

            # Create vault
            vault.create("test_password")
            assert vault.is_unlocked
            assert vault.exists

            # Lock and unlock
            vault.lock()
            assert not vault.is_unlocked

            assert vault.unlock("test_password")
            assert vault.is_unlocked

    def test_wrong_password(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test.enc"
            vault = CredentialVault(vault_path)

            vault.create("correct_password")
            vault.lock()

            assert not vault.unlock("wrong_password")
            assert not vault.is_unlocked

    def test_add_and_get_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test.enc"
            vault = CredentialVault(vault_path)
            vault.create("test_password")

            # Add credential
            cred = ProviderCredential(
                provider=ProviderType.LAMBDA_LABS,
                api_key="test_api_key_123",
            )
            vault.add(cred)

            # Retrieve
            retrieved = vault.get(ProviderType.LAMBDA_LABS)
            assert retrieved is not None
            assert retrieved.api_key == "test_api_key_123"

            # List
            providers = vault.list_providers()
            assert ProviderType.LAMBDA_LABS in providers

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test.enc"

            # Create and add
            vault1 = CredentialVault(vault_path)
            vault1.create("test_password")
            vault1.add(
                ProviderCredential(
                    provider=ProviderType.MISTRAL,
                    api_key="mistral_key",
                )
            )
            vault1.lock()

            # New instance should be able to read
            vault2 = CredentialVault(vault_path)
            assert vault2.unlock("test_password")

            cred = vault2.get(ProviderType.MISTRAL)
            assert cred is not None
            assert cred.api_key == "mistral_key"

    def test_remove_credential(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test.enc"
            vault = CredentialVault(vault_path)
            vault.create("test_password")

            vault.add(
                ProviderCredential(
                    provider=ProviderType.OPENAI,
                    api_key="openai_key",
                )
            )

            assert vault.has(ProviderType.OPENAI)
            assert vault.remove(ProviderType.OPENAI)
            assert not vault.has(ProviderType.OPENAI)

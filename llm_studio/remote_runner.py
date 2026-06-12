"""
Remote training execution over SSH.

Flow:
  1. Connect to the instance via paramiko
  2. Create a working directory on the remote
  3. SFTP-upload: llm_studio package + training data + config
  4. pip install dependencies (cached after first run)
  5. Run remote_train.py, streaming stdout back as log lines
  6. SFTP-download model artifacts back to local storage
  7. Return the local artifact path
"""
from __future__ import annotations

import io
import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Callable, Optional

import paramiko

logger = logging.getLogger(__name__)

# Path to the self-contained training script (shipped alongside this module)
_SCRIPT_PATH = Path(__file__).parent / "scripts" / "remote_train.py"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _build_client(host: str, port: int, username: str, key_path: Optional[str]) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict = dict(hostname=host, port=port, username=username, timeout=15)
    if key_path:
        connect_kwargs["key_filename"] = os.path.expanduser(key_path)
    client.connect(**connect_kwargs)
    return client


# ---------------------------------------------------------------------------
# Connection test
# ---------------------------------------------------------------------------

def test_connection(host: str, port: int, username: str, key_path: Optional[str]) -> tuple[bool, str]:
    """
    Return (success, message). Tests SSH connectivity and checks Python 3 + pip.
    """
    try:
        client = _build_client(host, port, username, key_path)
        _, stdout, stderr = client.exec_command("python3 --version && pip3 --version", timeout=10)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        client.close()
        if "Python 3" in out:
            return True, out
        return False, err or "Python 3 not found on remote"
    except paramiko.AuthenticationException:
        return False, "Authentication failed — check username and key"
    except paramiko.NoValidConnectionsError as e:
        return False, f"Cannot connect: {e}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Remote training
# ---------------------------------------------------------------------------

class RemoteRunner:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        key_path: Optional[str],
        log_callback: Callable[[str], None],
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.key_path = key_path
        self._log = log_callback

    def run(
        self,
        job_id: int,
        training_data: list[dict],
        config_dict: dict,
        local_output_dir: str,
    ) -> str:
        """
        Run training on the remote. Returns the local path where artifacts
        were downloaded to.
        """
        client = _build_client(self.host, self.port, self.username, self.key_path)
        sftp = client.open_sftp()
        remote_dir = f"/tmp/llm-studio-job-{job_id}"

        try:
            self._log(f"Connected to {self.host}")

            # 1. Create working directory
            self._exec(client, f"mkdir -p {remote_dir}/llm_studio/scripts")
            self._log(f"Remote working dir: {remote_dir}")

            # 2. Upload training data + config
            self._upload_text(sftp, f"{remote_dir}/data.json", json.dumps(training_data))
            self._upload_text(sftp, f"{remote_dir}/config.json", json.dumps(config_dict))
            self._log(f"Uploaded {len(training_data)} training samples and config")

            # 3. Upload training script
            self._upload_file(sftp, _SCRIPT_PATH, f"{remote_dir}/remote_train.py")
            self._log("Uploaded training script")

            # 4. Install dependencies (once — subsequent runs use pip cache)
            self._log("Checking dependencies on remote…")
            self._stream(
                client,
                "pip3 install torch transformers accelerate --quiet --disable-pip-version-check 2>&1",
            )

            # 5. Run training
            self._log("Starting remote training…")
            exit_code = self._stream(
                client,
                f"cd {remote_dir} && python3 remote_train.py "
                f"--data data.json --config config.json --output ./artifacts 2>&1",
            )
            if exit_code != 0:
                raise RuntimeError(f"Remote training exited with code {exit_code}")

            # 6. Download artifacts
            remote_artifact_dir = f"{remote_dir}/artifacts"
            os.makedirs(local_output_dir, exist_ok=True)
            self._download_dir(sftp, remote_artifact_dir, local_output_dir)
            self._log(f"Artifacts downloaded to {local_output_dir}")

            return local_output_dir

        finally:
            sftp.close()
            client.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _exec(self, client: paramiko.SSHClient, cmd: str) -> str:
        _, stdout, stderr = client.exec_command(cmd)
        stdout.channel.recv_exit_status()
        return stdout.read().decode()

    def _stream(self, client: paramiko.SSHClient, cmd: str) -> int:
        """Run a command and stream each output line to the log callback."""
        transport = client.get_transport()
        channel = transport.open_session()
        channel.exec_command(cmd)

        buffer = b""
        while not channel.exit_status_ready() or channel.recv_ready():
            if channel.recv_ready():
                chunk = channel.recv(4096)
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        self._log(f"[remote] {text}")
            else:
                time.sleep(0.1)

        if buffer.strip():
            self._log(f"[remote] {buffer.decode('utf-8', errors='replace').strip()}")

        return channel.recv_exit_status()

    def _upload_text(self, sftp: paramiko.SFTPClient, remote_path: str, content: str) -> None:
        with sftp.file(remote_path, "w") as f:
            f.write(content)

    def _upload_file(self, sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
        sftp.put(str(local_path), remote_path)

    def _download_dir(self, sftp: paramiko.SFTPClient, remote_dir: str, local_dir: str) -> None:
        try:
            entries = sftp.listdir_attr(remote_dir)
        except FileNotFoundError:
            logger.warning("Remote artifact dir not found: %s", remote_dir)
            return

        for entry in entries:
            remote_path = f"{remote_dir}/{entry.filename}"
            local_path = os.path.join(local_dir, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                os.makedirs(local_path, exist_ok=True)
                self._download_dir(sftp, remote_path, local_path)
            else:
                sftp.get(remote_path, local_path)

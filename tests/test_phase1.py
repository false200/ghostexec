"""Phase 1: scaffold, OpenEnv manifest, layout, and HTTP health surface."""

from pathlib import Path

import yaml
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def test_openenv_yaml_exists_and_metadata():
    path = ROOT / "openenv.yaml"
    assert path.is_file(), "openenv.yaml must exist at project root"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data.get("name") == "ghostexec"
    assert data.get("spec_version") == 1
    assert data.get("type") == "space"
    assert data.get("runtime") == "fastapi"
    assert data.get("app") == "server.app:app"
    desc = data.get("description")
    assert desc and isinstance(desc, str) and len(desc.strip()) > 0
    ver = data.get("version")
    assert ver and isinstance(ver, str) and len(ver.strip()) > 0


def test_expected_folder_structure():
    assert (ROOT / "models.py").is_file()
    assert (ROOT / "client.py").is_file()
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "server" / "app.py").is_file()
    assert (ROOT / "server" / "ghostexec_environment.py").is_file()
    assert (ROOT / "Dockerfile").is_file() or (ROOT / "server" / "Dockerfile").is_file()
    assert (ROOT / "server" / "requirements.txt").is_file()


def test_server_health_ping():
    from ghostexec.server.app import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "healthy"

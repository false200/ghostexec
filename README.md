---
title: Ghostexec Environment Server
emoji: 📢
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ghostexec Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Ghostexec environment is through the `GhostexecEnv` class:

```python
from ghostexec import GhostexecAction, GhostexecEnv

try:
    # Create environment from Docker image
    ghostexecenv = GhostexecEnv.from_docker_image("ghostexec-env:latest")

    # Reset
    result = ghostexecenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = ghostexecenv.step(GhostexecAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    ghostexecenv.close()
```

That's it! The `GhostexecEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t ghostexec-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

**HTTP vs WebSocket (episode state):** With the OpenEnv HTTP stack, each `POST /reset` and `POST /step` typically constructs a **new** environment instance for that request, so **back-to-back HTTP calls do not share one in-memory episode** (each `/step` only runs **one** action on that instance). Ghostexec **primes** an empty instance by loading the scenario before applying your action, so a lone `POST /step` still produces a real reward and `observation.metadata` (for example `step_ok`). For **many steps on the same running episode** (step 2, 3, … on unchanged state), use **`WebSocket /ws`**: open a connection, send reset, then send step messages on that same socket (or use a client that keeps a WebSocket session, for example `GhostexecEnv` over `base_url`).

## Environment Details

### Action
**GhostexecAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**GhostexecObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Ghostexec environment server running, you can connect directly:

```python
from ghostexec import GhostexecEnv

# Connect to existing server
ghostexecenv = GhostexecEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = ghostexecenv.reset()
result = ghostexecenv.step(GhostexecAction(message="Hello!"))
```

Note: When connecting to an existing server, `ghostexecenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from ghostexec import GhostexecAction, GhostexecEnv

# Connect with context manager (auto-connects and closes)
with GhostexecEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(GhostexecAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    GhostexecEnvironment,  # Pass class, not instance
    GhostexecAction,
    GhostexecObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from ghostexec import GhostexecAction, GhostexecEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with GhostexecEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(GhostexecAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/ghostexec_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

From the **`ghostexec`** directory (where `pyproject.toml` and `models.py` live):

```bash
# Recommended (matches installed package layout)
uv run uvicorn ghostexec.server.app:app --reload --host 0.0.0.0 --port 8000

# Same as many HF / OpenEnv docs (`models` + `server` on cwd)
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Or use the console script: `uv run server --port 8000`.

**CLI endpoint smoke** (no browser): in-process — `uv run python scripts/http_endpoint_smoke.py --local`  
Against a running server — `uv run python scripts/http_endpoint_smoke.py --url http://127.0.0.1:8000`  
Print example `curl` lines — `uv run python scripts/http_endpoint_smoke.py --print-curl`

## Complete stack test

Run the full suite (including HTTP/WebSocket/OpenEnv routes and a nested run of all other tests):

```bash
uv run pytest tests/ -q
```

With **`uvicorn` already running** on port 8000, hammer every HTTP edge case + WebSocket dead-ends:

```bash
uv run pytest tests/test_live_server_exhaustive.py -v --tb=short
```

Override base URL: `set GHOSTEXEC_LIVE_BASE_URL=http://127.0.0.1:9000` then the same command. If nothing is listening, all tests **skip**.

Optional live client check (real TCP WebSocket, not `TestClient`):

```bash
# Terminal 1
uv run server --port 8000
# Terminal 2
set GHOSTEXEC_WS_BASE_URL=http://127.0.0.1:8000
uv run pytest tests/test_complete_integration.py::test_ghostexec_env_client_against_live_url_if_set -q
```

## Phase 5 — training and Colab

- **Train locally (logs JSONL episode returns):**  
  `uv run python training/train.py --backend local --agent reinforce --episodes 30 --max-steps 14`  
  Use `--agent smart` for a scripted executive policy demo, or install optional LM stack: `uv sync --extra training`.
- **Colab (reward curve demo):** open `training/ghostexec_colab.ipynb` with the notebook working directory set to this `ghostexec` folder (so `pyproject.toml` is visible), then **Run All**.
- **Phase 5b — Unsloth / TRL post-training + GRPO (Colab, GPU):** open `training/ghostexec_unsloth_grpo_colab.ipynb`. It installs [Unsloth](https://github.com/unslothai/unsloth) + TRL, optional **SFT** on synthetic `(briefing → JSON action)` pairs, then **GRPO** with reward from a real `GhostexecEnvironment` first step (see [Unsloth RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)). Use a **GPU** runtime (T4+); default knobs are small so **Run All** can finish in one session—increase `GHOSTEXEC_GRPO_MAX_STEPS`, `GHOSTEXEC_SFT_SAMPLES`, etc. via environment variables documented in the first code cell. Set `GITHUB_REPO_URL` to your public git URL when the notebook is not already inside the repo. Helpers: `training/llm_action_parse.py`, `training/grpo_ghostexec_reward.py`.
- **Demo scenarios:** `scenarios/monday_morning.json`, `dinner_disaster.json`, `vip_meltdown.json` (+ `vip_meltdown_drift.json` for mood escalation).
- **HF blog paste-up:** see `training/HF_BLOG_DRAFT.md`.

## Project Structure

```
ghostexec/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── training/              # Phase 5: train.py, Colab notebooks (reward + Unsloth GRPO), HF blog draft
├── scenarios/             # World JSON (incl. monday_morning, dinner_disaster, vip_meltdown)
├── client.py              # GhostexecEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── ghostexec_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```

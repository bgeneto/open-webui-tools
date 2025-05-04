# Simple Math Tools API

An OpenAPI-compatible tool server providing mathematical computations (via SymPy) to LLMs. This server supports operations such as simplification, evaluation, solving (algebraic and differential equations), differentiation, integration, factoring, and expansion.

## Features

- Simplify expressions
- Numerically evaluate expressions with optional variable substitutions
- Solve algebraic equations
- Compute derivatives (with optional evaluation at a point)
- Compute integrals (indefinite and definite)
- Factor and expand expressions
- Solve ordinary differential equations (with optional initial conditions)
- OpenAPI schema and interactive docs via FastAPI (`/docs`)
- Open WebUI integration (v0.6+)

## Requirements

- Python 3.11 or higher

## Installation

```bash
# Clone the repository
git clone https://github.com/bgeneto/openapi-servers.git
cd openapi-servers/servers/simple_math

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\\Scripts\\activate  # Windows

# Install dependencies
pip install fastapi[all] sympy pydantic uvicorn
```

Alternatively, use a `requirements.txt`:

```text
fastapi[all]
sy mpy
pydantic
uvicorn
```

Then install:

```bash
pip install -r requirements.txt
```

## Running the Server

Start the FastAPI server on host `0.0.0.0` port `8000`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- OpenAPI schema: http://localhost:8000/openapi.json
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

Base Path: `/math`

| Path         | Method | Description                                  |
| ------------ | ------ | -------------------------------------------- |
| `/simplify`  | POST   | Simplify an expression                       |
| `/evaluate`  | POST   | Evaluate an expression with substitutions    |
| `/solve`     | POST   | Solve an algebraic equation                  |
| `/derivate`  | POST   | Compute a derivative                         |
| `/integrate` | POST   | Compute an integral (indefinite or definite) |
| `/factor`    | POST   | Factor an expression                         |
| `/expand`    | POST   | Expand an expression                         |
| `/dsolve`    | POST   | Solve a differential equation                |

See the interactive docs for request/response schemas and examples.

### Example: Simplify

```bash
curl -X POST http://localhost:8000/math/simplify \
  -H "Content-Type: application/json" \
  -d '{"expression": "x**2 + 2*x + 1", "variables": {"x": 2}}'
```

Response:

```json
{
  "success": true,
  "result": {
    "text_output": "9",
    "latex_output": "9"
  }
}
```

## Open WebUI Integration

Open WebUI (v0.6+) can connect to this tool server:

1. Launch the server (see **Running the Server**).
2. In Open WebUI, go to **Settings → Tools → + Add Tool**.
3. Enter the server URL (e.g., `http://localhost:8000`).
4. Save to discover all `/math/*` endpoints.
5. Use the tool icon in chat input to toggle math operations.

> If mounted via mcpo under a custom path, register each subpath explicitly, e.g.,:
> http://localhost:8000/math/simplify

## Configuration via mcpo

```yaml
tools:
  simple_math:
    url: http://localhost:8000/math
```

Launch with:

```bash
mcpo --config tools.yaml
```

## Troubleshooting & Tips

- Ensure the port is reachable (Docker, firewalls).
- For remote deployments, use HTTPS.
- Use interactive docs to test endpoints quickly.
- For production, consider Docker or a process manager.

---

Developed by bgeneto  •  Version 1.2.7  •  Last Modified: 2025-05-04
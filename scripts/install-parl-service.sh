#!/bin/bash
#
# Install PARL MCP Server as a systemd user service
#
# This creates a service that:
# - Starts automatically on login
# - Restarts on failure
# - Logs to journalctl
# - Uses environment variables from ~/.claude/.env
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_NAME="parl-mcp-server"
SERVICE_FILE="$HOME/.config/systemd/user/${SERVICE_NAME}.service"

echo "Installing PARL MCP Server as systemd user service..."

# Create systemd user directory if it doesn't exist
mkdir -p "$HOME/.config/systemd/user"

# Detect Python interpreter
PYTHON_BIN=$(which python3 || which python)
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Python not found. Install Python 3 first."
    exit 1
fi

# Detect conda/venv if active
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
    echo "✓ Using conda Python: $PYTHON_BIN"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    echo "✓ Using venv Python: $PYTHON_BIN"
else
    echo "✓ Using system Python: $PYTHON_BIN"
fi

# Create the service file
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=PARL Orchestrator MCP Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO_ROOT
ExecStart=$PYTHON_BIN $REPO_ROOT/examples/mcp/servers/parl_server.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment variables (edit these or source from ~/.claude/.env)
# Uncomment and customize as needed:
# Environment="PARL_ORCHESTRATOR_MODEL=deepinfra/Qwen/Qwen3-235B-A22B"
# Environment="PARL_SUB_AGENT_MODEL=deepinfra/Qwen/Qwen2.5-72B-Instruct"
# Environment="PARL_SYNTHESIS_MODEL=deepinfra/deepseek-ai/DeepSeek-V3.2"
# Environment="PARL_FACT_CHECK=true"
# Environment="PARL_DECOMPOSITION_BACKEND=litellm"
# Environment="MCP_PORT=8765"

# Load API keys from ~/.claude/.env if it exists
EnvironmentFile=-$HOME/.claude/.env

[Install]
WantedBy=default.target
EOF

echo "✓ Service file created: $SERVICE_FILE"

# Reload systemd user daemon
systemctl --user daemon-reload
echo "✓ Systemd daemon reloaded"

# Enable the service (start on login)
systemctl --user enable "$SERVICE_NAME.service"
echo "✓ Service enabled (will start on login)"

# Start the service now
systemctl --user start "$SERVICE_NAME.service"
echo "✓ Service started"

# Check status
echo ""
echo "Service status:"
systemctl --user status "$SERVICE_NAME.service" --no-pager || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ PARL MCP Server installed and running!"
echo ""
echo "Useful commands:"
echo "  Status:  systemctl --user status $SERVICE_NAME"
echo "  Logs:    journalctl --user -u $SERVICE_NAME -f"
echo "  Stop:    systemctl --user stop $SERVICE_NAME"
echo "  Start:   systemctl --user start $SERVICE_NAME"
echo "  Restart: systemctl --user restart $SERVICE_NAME"
echo "  Disable: systemctl --user disable $SERVICE_NAME"
echo ""
echo "Configuration:"
echo "  Edit: $SERVICE_FILE"
echo "  After editing, reload: systemctl --user daemon-reload && systemctl --user restart $SERVICE_NAME"
echo ""
echo "Server should be available at: http://localhost:8765"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

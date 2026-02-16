#!/bin/bash
#
# Uninstall PARL MCP Server systemd user service
#

set -e

SERVICE_NAME="parl-mcp-server"
SERVICE_FILE="$HOME/.config/systemd/user/${SERVICE_NAME}.service"

echo "Uninstalling PARL MCP Server service..."

# Stop the service if running
if systemctl --user is-active --quiet "$SERVICE_NAME.service"; then
    systemctl --user stop "$SERVICE_NAME.service"
    echo "✓ Service stopped"
fi

# Disable the service
if systemctl --user is-enabled --quiet "$SERVICE_NAME.service" 2>/dev/null; then
    systemctl --user disable "$SERVICE_NAME.service"
    echo "✓ Service disabled"
fi

# Remove the service file
if [ -f "$SERVICE_FILE" ]; then
    rm "$SERVICE_FILE"
    echo "✓ Service file removed"
fi

# Reload systemd
systemctl --user daemon-reload
echo "✓ Systemd daemon reloaded"

echo ""
echo "✅ PARL MCP Server service uninstalled"

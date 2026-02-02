#!/usr/bin/env bash
# Deploy duck (rio + heat equation) to ix.dev
# Usage: ./scripts/deploy-ix.sh
#
# Prerequisites:
#   - Docker running (colima start)
#   - ix.dev account configured
#
# This can also be triggered via the ix MCP server:
#   claude mcp add --transport http --scope user ix https://ix.dev/mcp
#   Then in Claude: "deploy duck to ix.dev"

set -euo pipefail

IMAGE="registry.ix.dev/bmorphism/duck:latest"

echo "Building duck docker image..."
docker build -t duck:latest .

echo "Tagging for ix.dev registry..."
docker tag duck:latest "$IMAGE"

echo "Pushing to ix.dev..."
docker push "$IMAGE"

echo "Creating ix.dev compute instance..."
# ix compute create "$IMAGE" --name duck --gpu
echo "NOTE: Run via ix MCP or ix CLI:"
echo "  ix compute create $IMAGE --name duck --gpu"

echo ""
echo "Done! Access at: https://duck.ix.dev"
echo "GitHub: https://github.com/bmorphism/duck-rio-heateq"

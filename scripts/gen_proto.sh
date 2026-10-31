#!/usr/bin/env bash
# Regenerates the Go and Python gRPC stubs from proto/paddy_agents.proto.
#
# Run this after ANY edit to the .proto file. The generated stubs are committed
# (they are force-added past the *_pb2.py rule in .gitignore), so a stale commit
# makes agents fail at import rather than at build.
#
# Prerequisites:
#   Python:  pip install "grpcio-tools>=1.78" "protobuf>=6.31.1,<7"
#   Go:      go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.12
#            go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.6.2
#            plus protoc itself on PATH
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Python stubs -> python/shared/"
python -m grpc_tools.protoc \
  -I proto \
  --python_out=python/shared \
  --grpc_python_out=python/shared \
  proto/paddy_agents.proto

echo "==> Go stubs -> proto/"
# paths=source_relative keeps output next to the .proto instead of nesting it
# under a github.com/... tree derived from go_package.
protoc \
  -I proto \
  --go_out=proto --go_opt=paths=source_relative \
  --go-grpc_out=proto --go-grpc_opt=paths=source_relative \
  proto/paddy_agents.proto

echo "==> Done. Verify with:"
echo "    PYTHONPATH=python/shared python -c 'import paddy_agents_pb2_grpc as g; print(g.LeafDiseaseAgentServiceStub)'"
echo "    go build ./..."

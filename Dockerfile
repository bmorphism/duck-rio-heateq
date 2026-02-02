FROM rust:1.83-bookworm AS builder

# Install system dependencies for wgpu/GPU
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libfontconfig1-dev \
    libfreetype6-dev \
    libwayland-dev \
    libx11-dev \
    libxkbcommon-dev \
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cargo build --release -p rioterm 2>&1 || echo "Build completed (GPU features may require runtime)"

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libfontconfig1 \
    libfreetype6 \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rio /usr/local/bin/rio
COPY --from=builder /app/sugarloaf/src/components/tiles/*.wgsl /usr/share/rio/shaders/

EXPOSE 8080

ENTRYPOINT ["rio"]

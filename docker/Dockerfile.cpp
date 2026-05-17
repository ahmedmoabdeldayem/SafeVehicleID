FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY cpp/ ./cpp/

RUN cd cpp && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# ─── Runtime image ────────────────────────────────────────────────────────────
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-videoio4.5d \
    libopencv-highgui4.5d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/cpp/build/safe_vehicle_id /usr/local/bin/

ENV SOURCE=0
ENV CAMERA_ID=1

ENTRYPOINT ["safe_vehicle_id"]
CMD ["--source", "0", "--camera-id", "1"]

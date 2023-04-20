FROM rust:1.68.2-slim-bullseye

WORKDIR /home

RUN apt-get update && \
    apt-get -y install \
        ca-certificates \
        cmake \
        pkg-config \
        musl-tools \
        wget \
        libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz && \
    tar zxf onnxruntime-linux-x64-1.14.1.tgz && \
    mv onnxruntime-linux-x64-1.14.1 /lib/onnxruntime

ENV ORT_LIB_LOCATION=/lib/onnxruntime
ENV ORT_STRATEGY=system

COPY . .
RUN cargo install --path .

EXPOSE 8080

CMD ["/usr/local/cargo/bin/proton"]

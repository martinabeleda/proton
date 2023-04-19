FROM rust:1.68.2-slim-bullseye as builder

# muslc is required in order to build the rust image.
RUN apt-get update && \
    apt-get -y install ca-certificates cmake musl-tools libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY . .
RUN cargo install --path .


FROM debian:bullseye-slim

COPY --from=builder /usr/local/cargo/bin/proton /usr/local/bin/proton

EXPOSE 8080

CMD ["/usr/local/bin/proton"]

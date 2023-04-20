.PHONY: cook-image start itest test clean

DOCKER_IMAGE = martinabeleda/proton
DOCKER_TAG = latest

cook-image:
	docker buildx build --platform linux/amd64 -t ${DOCKER_IMAGE}:${DOCKER_TAG} .

start: stop
	docker compose up -d

stop:
	docker compose down --remove-orphans

itest: start
	cargo run --bin client

test:
	cargo test

clean:
	rm -rf target/

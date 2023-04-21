# Proton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![main workflow](https://github.com/martinabeleda/proton/actions/workflows/rust.yml/badge.svg)

A rust based service for serving machine learning predictions.

## :pencil: Features

- Supports REST and gRPC
- Concurrency model: thread per model
- Supports per model queues via async channels
- Onnxruntime backend

## :construction: Dependencies

### Protbuf

In order to build the gRPC server, you'l need the `protoc` Protocol Buffers compiler. See [the tonic documentation](https://github.com/hyperium/tonic#dependencies) to set this up.

### Onnxruntime

Onnxruntime is the engine we use to power inference. See [onnxruntime docs](https://onnxruntime.ai/index.html#getStartedTable) for more complete installation instructions.

#### MacOS

```shell
brew install onnxruntime
```

Export the lib path:

```shell
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=$(brew --prefix onnxruntime)
```

#### Linux

On Linux, you'll need to download the tarball

```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz
tar zxf onnxruntime-linux-x64-1.14.1.tgz
mv onnxruntime-linux-x64-1.14.1 /lib/onnxruntime
```

```shell
export ORT_STRATEGY=system
export ORT_LIB_LOCATION=/lib/onnxruntime
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/onnxruntime/lib
```

## :test_tube: Testing

Download the test models:

```shell
curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
curl -LO "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx"
```

Start the service

```shell
cargo run
```

See the deployed models:

```shell
curl -X GET -H "Content-Type: application/json" http://localhost:8080/models
```

The client script runs many concurrent requests against the server and logs the elapsed for each
request and the total time. To run the client:

```shell
❯ cargo run --bin client
2023-04-19T18:31:32.023177Z  INFO client: src/bin/client.rs:112: Sending request for "squeezenet"
2023-04-19T18:31:32.026326Z  INFO client: src/bin/client.rs:112: Sending request for "maskrcnn"
...
2023-04-19T18:31:32.910031Z  INFO client: src/bin/client.rs:75: Success for model "squeezenet", prediction_id: 1ff1417e-5b8e-4fa7-8aff-5111fa50f3ac
2023-04-19T18:31:33.722041Z  INFO client: src/bin/client.rs:75: Success for model "maskrcnn", prediction_id: 373fdc2c-8455-4b54-b7a5-f910ba1c506c
...
2023-04-19T18:32:00.538712Z  INFO client: src/bin/client.rs:143: Model: "squeezenet"
2023-04-19T18:32:00.538725Z  INFO client: src/bin/client.rs:144: Total time: 13.100649499s
2023-04-19T18:32:00.538734Z  INFO client: src/bin/client.rs:145: Average time: 524.025979ms
2023-04-19T18:32:00.538742Z  INFO client: src/bin/client.rs:146: Median time: 551.289125ms
2023-04-19T18:32:00.538750Z  INFO client: src/bin/client.rs:147: p95 time: 630.9305ms
2023-04-19T18:32:00.538759Z  INFO client: src/bin/client.rs:148: p99 time: 680.362292ms
...
2023-04-19T18:32:00.538772Z  INFO client: src/bin/client.rs:143: Model: "maskrcnn"
2023-04-19T18:32:00.538778Z  INFO client: src/bin/client.rs:144: Total time: 372.399760668s
2023-04-19T18:32:00.538784Z  INFO client: src/bin/client.rs:145: Average time: 14.895990426s
2023-04-19T18:32:00.538789Z  INFO client: src/bin/client.rs:146: Median time: 14.996715167s
2023-04-19T18:32:00.538795Z  INFO client: src/bin/client.rs:147: p95 time: 27.056427083s
2023-04-19T18:32:00.538801Z  INFO client: src/bin/client.rs:148: p99 time: 28.186501209s
```

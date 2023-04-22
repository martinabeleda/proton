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
cargo run --bin client
```

We also provide the same client for gRPC:

```shell
cargo run --bin grpc-client
```

## :bench: Benchmark

Running locally on an M1 Macbook pro, gRPC performs better. The gap is marginal for MaskRCNN where compute
is a much larger proportion of the response time

| Client Type | Model      | Total Time    | Average Time | Median Time  | p95 Time      | p99 Time      |
| ----------- | ---------- | ------------- | ------------ | ------------ | ------------- | ------------- |
| client      | squeezenet | 1.225398832s  | 136.155425ms | 143.143708ms | 198.443083ms  | 198.443083ms  |
| grpc-client | squeezenet | 182.413958ms  | 18.241395ms  | 17.662917ms  | 20.213208ms   | 20.213208ms   |
| client      | maskrcnn   | 76.612492085s | 6.964772007s | 6.975712042s | 11.401473875s | 12.505639542s |
| grpc-client | maskrcnn   | 57.749369832s | 5.774936983s | 5.363160166s | 10.23204925s  | 10.23204925s  |

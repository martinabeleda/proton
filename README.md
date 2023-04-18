# proton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![main workflow](https://github.com/martinabeleda/proton/actions/workflows/rust.yml/badge.svg)

A rust based service for serving machine learning predictions.

## Install `onnxruntime`

On MacOS:

```shell
brew install onnxruntime
```

Export the lib path:

```shell
export ONNX_RUNTIME_LIB_DIR=$(brew --prefix onnxruntime)
```

## Testing

Download the test model:

```shell
curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"
curl -LO "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx"
curl -LO "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx"
```

Start the service

```shell
cargo run --bin server
```

The client script runs many concurrent requests against the server and logs the elapsed for each
request and the total time. To run the client:

```shell
❯ cargo run --bin client
2023-04-11T02:34:51.750727Z  INFO client: src/client.rs:61: Input array: [1, 3, 224, 224]
2023-04-11T02:34:52.192633Z  INFO client: src/client.rs:32: Success e1c0321f-1be8-4436-b5cf-d23ba9fd4976
...
2023-04-18T22:02:58.898530Z  INFO client: src/bin/client.rs:93: Total time: 1.831743333s
2023-04-18T22:02:58.898539Z  INFO client: src/bin/client.rs:94: Average time: 1.028013755s
2023-04-18T22:02:58.898547Z  INFO client: src/bin/client.rs:95: p95 time: 1.590004666s
2023-04-18T22:02:58.898553Z  INFO client: src/bin/client.rs:96: p99 time: 1.730717125s
```

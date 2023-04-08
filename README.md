# proton

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

Start the service

```shell
cargo run
```

Make a prediction

```shell
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "squeezenet", "input_data": [0.0, 1.0, 2.0]}' http://localhost:8080/predict
```

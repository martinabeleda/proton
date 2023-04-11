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
2023-04-11T02:34:52.954031Z  INFO client: src/client.rs:32: Success fc2078e3-b3a6-4a6c-a5b4-bad093c336a5
2023-04-11T02:34:52.954101Z  INFO client: src/client.rs:93: Elapsed times: [392.465833ms, 407.864958ms, 429.669375ms, 444.556917ms, 447.875708ms, 449.03975ms, 451.535584ms, 464.551834ms, 467.682167ms, 469.597833ms, 477.078291ms, 490.135875ms, 491.900667ms, 497.610084ms, 508.749292ms, 525.443666ms, 542.4995ms, 561.1515ms, 591.543292ms, 591.913541ms, 618.668416ms, 633.163208ms, 659.706167ms, 660.14775ms, 665.351291ms, 683.048083ms, 683.565458ms, 697.303833ms, 705.694542ms, 706.001ms, 718.338ms, 727.04025ms, 736.476625ms, 763.996208ms, 770.853416ms, 807.890292ms, 821.219458ms, 836.138458ms, 838.330584ms, 842.228875ms, 857.975375ms, 859.833875ms, 862.920708ms, 870.914875ms, 872.118042ms, 907.068042ms, 918.910834ms, 921.350667ms, 929.861166ms, 933.313708ms]
2023-04-11T02:34:52.954128Z  INFO client: src/client.rs:98: Total time: 1.201856958s
2023-04-11T02:34:52.954135Z  INFO client: src/client.rs:99: Average time: 664.245897ms
2023-04-11T02:34:52.954142Z  INFO client: src/client.rs:100: p95 time: 921.350667ms
2023-04-11T02:34:52.954148Z  INFO client: src/client.rs:101: p99 time: 933.313708ms
```

# proton

A rust based service for serving machine learning predictions

## Testing

```shell
cargo run
```

```shell
curl -X POST -H "Content-Type: application/json" -d '{"model_type": "onnx", "model_path": "/path/to/your/model.onnx", "input_data": [0.0, 1.0, 2.0]}' http://localhost:8080/infer
```

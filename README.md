# proton

A rust based service for serving machine learning predictions

## Testing

Start the service

```shell
cargo run
```

Make a prediction

```shell
curl -X POST -H "Content-Type: application/json" -d '{"model_name": "squeezenet", "input_data": [0.0, 1.0, 2.0]}' http://localhost:8080/infer
```

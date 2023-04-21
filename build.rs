fn main() {
    tonic_build::configure()
        .build_server(true)
        .compile(&["proto/predict.proto"], &["proto"])
        .unwrap();
}

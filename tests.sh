export RUST_BACKTRACE=1
cargo llvm-cov --all-features --lcov --output-path ./target/lcov.info -- --nocapture
cargo test --no-default-features
cargo test --no-default-features --features rational
cargo test --no-default-features --features rational,quaternion,libm,bytemuck,serde
cargo test --all-features
cargo test
cargo test --features rational,quaternion,bytemuck,serde

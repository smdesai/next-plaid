//! Bindings generator, built from the same `uniffi` version as the library so
//! the generated Swift scaffolding matches the runtime. Invoked in the
//! XCFramework build as: `uniffi-bindgen generate --library <lib> --language swift`.
fn main() {
    uniffi::uniffi_bindgen_main()
}

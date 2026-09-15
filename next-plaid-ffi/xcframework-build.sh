#!/usr/bin/env bash
#
# Build NextPlaidFFI.xcframework (macOS + iOS device + iOS simulator) and the
# matching UniFFI Swift bindings from the next-plaid-ffi crate.
#
# Output (under ./build):
#   NextPlaidFFI.xcframework   -> Swift package binaryTarget
#   next_plaid_ffi.swift       -> the NextPlaidBindings source (imports module next_plaid_ffiFFI)
#
# The clang module stays UniFFI's default name `next_plaid_ffiFFI` so the
# generated Swift's `import next_plaid_ffiFFI` matches the module.modulemap.
# The .xcframework file itself is named NextPlaidFFI for the SPM binaryTarget.
#
# Env overrides:
#   FEATURES   cargo features for the static libs (default: accelerate)
#   PROFILE    cargo profile (default: release)
set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CRATE_DIR"

LIB_NAME="next_plaid_ffi"           # cargo [lib] name -> libnext_plaid_ffi.a
STATIC_LIB="lib${LIB_NAME}.a"
FRAMEWORK_NAME="NextPlaidFFI"
PROFILE="${PROFILE:-release}"
FEATURES="${FEATURES:-accelerate}"

BUILD_DIR="$CRATE_DIR/build"
BINDINGS_DIR="$BUILD_DIR/bindings"
HEADERS_DIR="$BUILD_DIR/headers"

# macOS is Apple Silicon only (add x86_64-apple-darwin back for Intel Macs).
# iOS device is arm64-only. The simulator is arm64-only here; add
# x86_64-apple-ios to IOS_SIM_ARCHS to also support the simulator on Intel Macs.
MAC_ARCHS=(aarch64-apple-darwin)
IOS_DEVICE=aarch64-apple-ios
IOS_SIM_ARCHS=(aarch64-apple-ios-sim)

ALL_TARGETS=("${MAC_ARCHS[@]}" "$IOS_DEVICE" "${IOS_SIM_ARCHS[@]}")

# Minimum OS versions must match the consuming Swift package (Plaid: iOS 17 /
# macOS 14). Rust/cc embed these into each Mach-O's LC_BUILD_VERSION via the
# standard deployment-target env vars; without them the objects default to the
# host SDK and the linker warns about a version mismatch.
MACOS_MIN="${MACOS_MIN:-14.0}"
IOS_MIN="${IOS_MIN:-17.0}"

lib_path() { echo "$CRATE_DIR/target/$1/$PROFILE/$STATIC_LIB"; }

echo "==> Ensuring rust targets are installed"
for t in "${ALL_TARGETS[@]}"; do rustup target add "$t" >/dev/null; done

feature_args=()
[ -n "$FEATURES" ] && feature_args=(--features "$FEATURES")

echo "==> Building static libraries (profile=$PROFILE features='$FEATURES')"
for t in "${ALL_TARGETS[@]}"; do
  echo "    - $t"
  case "$t" in
    *-apple-darwin) dep_env=(env "MACOSX_DEPLOYMENT_TARGET=$MACOS_MIN") ;;
    *-apple-ios*)   dep_env=(env "IPHONEOS_DEPLOYMENT_TARGET=$IOS_MIN") ;;
    *)              dep_env=(env) ;;
  esac
  "${dep_env[@]}" cargo build --profile "$PROFILE" --target "$t" "${feature_args[@]}"
done

echo "==> Assembling per-platform static libs"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/macos" "$BUILD_DIR/ios" "$BUILD_DIR/ios-sim"

MAC_LIBS=(); for a in "${MAC_ARCHS[@]}"; do MAC_LIBS+=("$(lib_path "$a")"); done
MAC_FAT="$BUILD_DIR/macos/$STATIC_LIB"
lipo -create "${MAC_LIBS[@]}" -output "$MAC_FAT"

IOS_DEV_LIB="$BUILD_DIR/ios/$STATIC_LIB"
cp "$(lib_path "$IOS_DEVICE")" "$IOS_DEV_LIB"

SIM_LIBS=(); for a in "${IOS_SIM_ARCHS[@]}"; do SIM_LIBS+=("$(lib_path "$a")"); done
IOS_SIM_LIB="$BUILD_DIR/ios-sim/$STATIC_LIB"
if [ "${#SIM_LIBS[@]}" -gt 1 ]; then
  lipo -create "${SIM_LIBS[@]}" -output "$IOS_SIM_LIB"
else
  cp "${SIM_LIBS[0]}" "$IOS_SIM_LIB"
fi

# Strip DWARF debug symbols from each assembled archive. This keeps every
# global/external symbol the linker needs (the C-ABI FFI exports and all
# referenced code) while dropping debug info, which is dead weight in a shipped
# static lib. Cuts each slice by ~10%.
echo "==> Stripping debug symbols from static libs"
for lib in "$MAC_FAT" "$IOS_DEV_LIB" "$IOS_SIM_LIB"; do
  strip -S "$lib"
done

echo "==> Generating Swift bindings (uniffi-bindgen, matching crate version)"
mkdir -p "$BINDINGS_DIR"
# bindgen runs on the host; metadata is target-independent so any built lib works.
cargo run --profile "$PROFILE" --bin uniffi-bindgen -- \
  generate --library "$(lib_path "${MAC_ARCHS[0]}")" --language swift --out-dir "$BINDINGS_DIR"

echo "==> Assembling headers + modulemap"
mkdir -p "$HEADERS_DIR"
cp "$BINDINGS_DIR/${LIB_NAME}FFI.h" "$HEADERS_DIR/"
# XCFramework/SPM expects the clang module map to be named module.modulemap.
cp "$BINDINGS_DIR/${LIB_NAME}FFI.modulemap" "$HEADERS_DIR/module.modulemap"

echo "==> Creating $FRAMEWORK_NAME.xcframework"
XCF="$BUILD_DIR/$FRAMEWORK_NAME.xcframework"
rm -rf "$XCF"
xcodebuild -create-xcframework \
  -library "$MAC_FAT"     -headers "$HEADERS_DIR" \
  -library "$IOS_DEV_LIB" -headers "$HEADERS_DIR" \
  -library "$IOS_SIM_LIB" -headers "$HEADERS_DIR" \
  -output "$XCF"

cp "$BINDINGS_DIR/${LIB_NAME}.swift" "$BUILD_DIR/"

echo ""
echo "==> Done."
echo "    XCFramework : $XCF"
echo "    Swift binding: $BUILD_DIR/${LIB_NAME}.swift"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${ANDROID_PLATFORM_JAR:?Set ANDROID_PLATFORM_JAR to android-35/android.jar}"
: "${ANDROID_BUILD_TOOLS:?Set ANDROID_BUILD_TOOLS to build-tools/35.0.0}"
mkdir -p build/classes build/dex out
"$ANDROID_BUILD_TOOLS/aapt2" compile --dir res -o build/resources.zip
"$ANDROID_BUILD_TOOLS/aapt2" link -o build/resources.apk -I "$ANDROID_PLATFORM_JAR" --manifest AndroidManifest.xml --min-sdk-version 26 --target-sdk-version 35 -A assets build/resources.zip
if command -v javac >/dev/null; then
  javac -encoding UTF-8 -source 8 -target 8 -classpath "$ANDROID_PLATFORM_JAR" -d build/classes src/jp/tanakabutton/videonotes/*.java
else
  : "${ECJ_JAR:?Set ECJ_JAR to Eclipse ecj.jar if javac is unavailable}"
  java -jar "$ECJ_JAR" -encoding UTF-8 -8 -classpath "$ANDROID_PLATFORM_JAR" -d build/classes src/jp/tanakabutton/videonotes/*.java
fi
"$ANDROID_BUILD_TOOLS/d8" --lib "$ANDROID_PLATFORM_JAR" --min-api 26 --output build/dex build/classes/jp/tanakabutton/videonotes/*.class
cp build/resources.apk build/unsigned.apk
(cd build/dex && zip -q ../unsigned.apk classes*.dex)
"$ANDROID_BUILD_TOOLS/zipalign" -f -p 4 build/unsigned.apk build/aligned.apk
: "${APK_KEYSTORE:?Set APK_KEYSTORE to the signing keystore path}"
: "${APK_KEY_ALIAS:=video-notes}"
: "${APK_KEY_PASSWORD:?Set APK_KEY_PASSWORD to the keystore password}"
"$ANDROID_BUILD_TOOLS/apksigner" sign --ks "$APK_KEYSTORE" --ks-key-alias "$APK_KEY_ALIAS" --ks-pass env:APK_KEY_PASSWORD --key-pass env:APK_KEY_PASSWORD --out out/video-notes-1.2.0.apk build/aligned.apk
"$ANDROID_BUILD_TOOLS/apksigner" verify --verbose out/video-notes-1.2.0.apk

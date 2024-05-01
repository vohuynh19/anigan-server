#!/bin/bash

admin_sdk_url="https://firebasestorage.googleapis.com/v0/b/xetpasta.appspot.com/o/adminSdk.json?alt=media&token=97f88d31-98e2-4d37-999b-2697554944ea"
model_url="https://firebasestorage.googleapis.com/v0/b/xetpasta.appspot.com/o/face_paint_512_v2.pt?alt=media"

admin_sdk_path="adminSdk.json"
model_path="face_paint_512_v2.pt"

curl -o "$admin_sdk_path" "$admin_sdk_url"
curl -o "$model_path" "$model_url"



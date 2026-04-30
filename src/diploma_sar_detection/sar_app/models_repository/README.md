# ONNX model repository

Place exported `.onnx` models and matching `metadata.json` files in this directory.

Example:

```json
{
  "name": "YOLO26n SAR 1024",
  "path": "best_yolo26n_1024.onnx",
  "input_sizes": [1024],
  "class_names": [
    "swimmer",
    "boat",
    "jetski",
    "life_saving_appliances",
    "buoy"
  ],
  "description": "YOLO26n model exported to ONNX for SAR object detection."
}
```

The Tkinter application reads only ONNX files through `onnxruntime`; `.pt` checkpoints are used only during export.


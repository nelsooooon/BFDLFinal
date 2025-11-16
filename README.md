# BFDLFinal – Multi-Format TensorFlow Image Classifier

This project provides a trained image classification model exported to three deployment targets: native TensorFlow SavedModel, TensorFlow Lite (for mobile/edge), and TensorFlow.js (for browser inference). It predicts one of the classes defined in `tflite/label.txt` (e.g. `glacier`, `sea`, `street`).

## Key Features

- Multi-format exports: SavedModel, TFLite, TFJS
- Single source of truth for labels (`tflite/label.txt`)
- Ready for edge, server, and browser inference
- Minimal setup: dependencies pinned in `requirements.txt`
- Example Python inference snippets (SavedModel & TFLite)
- Easily servable TFJS model (`tfjs_model/model.json`)

## Repository Structure

```
saved_model/          # TensorFlow SavedModel directory
	saved_model.pb
	fingerprint.pb
	variables/
		variables.index
		variables.data-00000-of-00001
tflite/
	model.tflite        # TensorFlow Lite flatbuffer
	label.txt           # Class labels
tfjs_model/
	model.json          # TFJS model graph + weights manifest
notebook.ipynb        # Exploration / experimentation notebook
requirements.txt      # Python dependencies
```

## Technology / Stack

- Python (tested with 3.8+ recommendation)
- TensorFlow (`tensorflow==2.19.0`)
- TensorFlow Lite runtime (included in TensorFlow package)
- TensorFlow.js (for browser usage)
- NumPy, OpenCV, Pillow (preprocessing / image loading)
- Pandas, Matplotlib, Seaborn, scikit-learn (analysis & experimentation – optional at inference time)
- Keras Tuner (used during model development)

Only TensorFlow, NumPy, Pillow (or OpenCV) are strictly required for basic inference.

## Installation

Clone and set up a virtual environment (recommended):

```bash
git clone https://github.com/nelsooooon/BFDLFinal.git
cd BFDLFinal

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Confirm TensorFlow is installed:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Labels

The model outputs indices mapped to human-readable class names stored line-by-line in `tflite/label.txt`.

```python
with open("tflite/label.txt") as f:
		classes = [line.strip() for line in f if line.strip()]
```

## Inference (TensorFlow SavedModel)

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

saved_model_path = "saved_model"  # directory containing saved_model.pb
image_path = "image.jpg"  # replace with your test image

# Load labels
with open("tflite/label.txt") as f:
		classes = [l.strip() for l in f if l.strip()]

# Preprocess image (adjust target_size to training size if different)
img = load_img(image_path, target_size=(128, 128))
arr = img_to_array(img)
arr = np.expand_dims(arr, axis=0) / 255.0

# Wrap SavedModel
layer = tf.keras.layers.TFSMLayer(saved_model_path, call_endpoint="serving_default")
pred = layer(arr)

# Inspect available output keys if unsure
print("Output keys:", pred.keys())

probs = pred["output_0"].numpy()[0]  # adjust key if different
pred_class = classes[int(np.argmax(probs))]

print("Probabilities:", probs)
print("Predicted class:", pred_class)
```

## Inference (TensorFlow Lite)

```python
import numpy as np
import tensorflow as tf
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = Image.open("image.jpg").resize((128, 128))  # adapt size as needed
arr = np.array(img, dtype=np.float32) / 255.0
arr = np.expand_dims(arr, axis=0)

interpreter.set_tensor(input_details[0]["index"], arr)
interpreter.invoke()
probs = interpreter.get_tensor(output_details[0]["index"])[0]

with open("tflite/label.txt") as f:
		classes = [l.strip() for l in f if l.strip()]

print("Probabilities:", probs)
print("Predicted class:", classes[int(np.argmax(probs))])
```

## Inference (TensorFlow.js)

Serve the `tfjs_model/` directory locally (e.g. simple static server):

```bash
python -m http.server 8000
```

Example HTML snippet:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<img id="input" src="image.jpg" crossorigin="anonymous" />
<script>
	async function run() {
		const model = await tf.loadLayersModel('http://localhost:8000/tfjs_model/model.json');
		const img = document.getElementById('input');
		let tensor = tf.browser.fromPixels(img)
			.resizeBilinear([128, 128])
			.toFloat()
			.div(255)
			.expandDims();
		const preds = model.predict(tensor).dataSync();
		const labels = ["glacier", "sea", "street"]; // or fetch dynamically
		const idx = preds.indexOf(Math.max(...preds));
		console.log('Predicted:', labels[idx], preds);
	}
	run();
</script>
```

## Notebook Usage

Use `notebook.ipynb` to experiment with preprocessing steps, compare outputs between SavedModel and TFLite, or visualize predictions.

## Important Files

- `saved_model/` – TensorFlow SavedModel directory
- `tflite/model.tflite` – TensorFlow Lite flatbuffer
- `tflite/label.txt` – Class labels
- `tfjs_model/model.json` – TensorFlow.js model (graph + weights manifest)
- `requirements.txt` – Full dependency list (includes tools used during development)

## Troubleshooting

- Output key mismatch: print `pred.keys()` to verify the correct tensor name.
- Shape errors: ensure input resize matches training dimension (e.g. 128×128).
- Performance issues in TFJS: call `tf.setBackend('webgl')` if not default.
- Missing GPU drivers: inference still works on CPU for this small model.

## Future Improvements

- Add conversion scripts (SavedModel → TFLite / TFJS)
- Provide training pipeline & dataset documentation
- Add benchmark comparisons (latency, size, accuracy) per format
- Include automated tests for inference consistency across formats

## License

Add your license information here (e.g. MIT, Apache 2.0). If none specified, usage may be unclear—consider adding one.

## Disclaimer

This repository contains inference artifacts only. Training data and full training scripts are not included.

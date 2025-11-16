# BFDLFinal

This repository contains a trained machine learning model (TensorFlow) exported to multiple formats for inference: SavedModel (TensorFlow), TensorFlow Lite (TFLite), and TensorFlow.js (TFJS).

Key features:

- Trained model exported to multiple formats: SavedModel, TFLite, and TFJS
- Example labels for the classifier (`tflite/label.txt`)
- Python dependencies listed in `requirements.txt`

Technology / Stack:

- Python 3.8+ (recommended)
- TensorFlow (`tensorflow==2.19.0` as listed in `requirements.txt`)
- TensorFlow Lite (for mobile/edge inference)
- TensorFlow.js (for browser inference)
- OpenCV, Pillow (image processing)

Using the SavedModel (TensorFlow)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

img = load_img("image.jpg", target_size=(128,128))

img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255

saved_model = tf.keras.layers.TFSMLayer(saved_model_path, call_endpoint='serving_default')
prediction = saved_model(img)

print(prediction)

class_prediction = classes[np.argmax(prediction['output_0'].numpy()[0])]

print(f"\nHasil Prediksi: {class_prediction}")
```

Important files in the repo:

- `saved_model/` — TensorFlow SavedModel
- `tflite/model.tflite` — TFLite model
- `tflite/label.txt` — output labels (example: `glacier`, `sea`, `street`)
- `tfjs_model/model.json` — TFJS model for browser
- `requirements.txt` — Python dependency list

from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import cv2
import numpy as np
import os


# ============================================================
# CUSTOM MAMBA-STYLE LAYER
# ============================================================

class LightweightSSMLayer(keras.layers.Layer):

    def __init__(self, dim, state_dim, **kwargs):
        super().__init__(**kwargs)

        self.dim = dim
        self.state_dim = state_dim

    def build(self, input_shape):

        self.A = self.add_weight(
            shape=(self.state_dim, self.state_dim),
            initializer="random_normal",
            trainable=True,
            name="state_matrix"
        )

        self.B = keras.layers.Dense(
            self.state_dim,
            name="input_to_state"
        )

        self.C = keras.layers.Dense(
            self.dim,
            name="state_to_output"
        )

    def call(self, x):

        state = keras.ops.zeros_like(
            self.B(x[:, 0, :])
        )

        outputs = []

        for t in range(x.shape[1]):

            current = self.B(x[:, t, :])

            state = keras.ops.tanh(
                keras.ops.matmul(state, self.A) + current
            )

            outputs.append(
                self.C(state)
            )

        return keras.ops.stack(
            outputs,
            axis=1
        )


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "baseline_mamba_model.keras"
)


print("Loading posture model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "LightweightSSMLayer": LightweightSSMLayer
    },
    compile=False
)

print("Posture model loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = {
    0: "Bad Sitting Posture",
    1: "Good Sitting Posture"
}


# ============================================================
# POSTURE PREDICTION FUNCTION
# ============================================================

def predict_posture(frame):

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Resize to model input size
    frame_resized = cv2.resize(
        frame_rgb,
        (224, 224)
    )

    # Normalize
    frame_input = (
        frame_resized.astype("float32")
        / 255.0
    )

    # Add batch dimension
    frame_input = np.expand_dims(
        frame_input,
        axis=0
    )

    # Run model
    prediction = model.predict(
        frame_input,
        verbose=0
    )

    # Get predicted class
    predicted_class = int(
        np.argmax(prediction[0])
    )

    # Get confidence
    confidence = float(
        prediction[0][predicted_class]
    )

    # Get class name
    posture = class_names[
        predicted_class
    ]

    return posture, confidence


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SERVER TEST
# ============================================================

@app.route("/test")
def test():

    return jsonify({
        "status": "working",
        "model_loaded": model is not None
    })


# ============================================================
# POSTURE PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # Check image
        if "image" not in request.files:

            return jsonify({
                "error": "No image received"
            }), 400


        # Read uploaded image
        image_file = request.files["image"]

        image_bytes = image_file.read()


        # Convert bytes → NumPy array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        # Decode image
        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        # Check image
        if frame is None:

            return jsonify({
                "error": "Could not read image"
            }), 400


        # Run posture prediction
        posture, confidence = predict_posture(
            frame
        )


        # Return result
        return jsonify({

            "posture": posture,

            "confidence": confidence

        })


    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        return jsonify({

            "error": str(error)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

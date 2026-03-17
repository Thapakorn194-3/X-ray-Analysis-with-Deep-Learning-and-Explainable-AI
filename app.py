from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input as dn_preprocess  # ✅ FIX
import cv2
import base64
from io import BytesIO
from PIL import Image
import os
from skimage.segmentation import slic, mark_boundaries
from sklearn.linear_model import LinearRegression
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app)

# ─────────────────────────────────────────────
# CONFIGURATION — แก้ path ตรงนี้
# ─────────────────────────────────────────────
MODEL_PATHS = {
    'cnn': r'D:\X-ray-Analysis-with-Deep-Learning-and-Explainable-AI\exp2_best.keras',
    'svm': r'D:\X-ray-Analysis-with-Deep-Learning-and-Explainable-AI\svm1_best.pkl',
    'xgb': r'D:\X-ray-Analysis-with-Deep-Learning-and-Explainable-AI\xgb1_best.json',
}

INPUT_SIZE        = (320, 320)
CLASS_NAMES       = ['Normal', 'Tuberculosis']
LAST_CONV_LAYER   = 'conv5_block16_concat'   # ✅ FIX: hardcode ให้ตรงกับ Colab


# ══════════════════════════════════════════════
# SECTION 1 — CUSTOM LOSS
# ══════════════════════════════════════════════
@tf.keras.utils.register_keras_serializable(package="losses")
class BinaryFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha_neg=1.0, alpha_pos=0.75,
                 reduction='sum_over_batch_size', name="binary_focal_loss", **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)
        self.gamma     = float(gamma)
        self.alpha_neg = float(alpha_neg)
        self.alpha_pos = float(alpha_pos)

    def call(self, y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        bce    = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
        p      = tf.sigmoid(y_pred_logits)
        p_t    = y_true * p + (1 - y_true) * (1 - p)
        focal  = tf.pow(1 - p_t, self.gamma) * bce
        a_t    = y_true * self.alpha_pos + (1 - y_true) * self.alpha_neg
        return tf.reduce_mean(a_t * focal)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha_neg": self.alpha_neg, "alpha_pos": self.alpha_pos})
        return cfg


# ══════════════════════════════════════════════
# SECTION 2 — LOAD MODELS
# ══════════════════════════════════════════════
print("\n" + "="*60)
print("Loading models...")

# --- CNN ---
try:
    cnn_model = keras.models.load_model(
        MODEL_PATHS['cnn'],
        custom_objects={"binary_focal_loss": BinaryFocalLoss},
        compile=False,
    )
    print(f"✅ CNN loaded | input: {cnn_model.input_shape}")

    # Feature extractor: หา GAP layer อัตโนมัติ (ตรงกับ lime_xgboost.py)
    _feature_layer = None
    for layer in cnn_model.layers:
        if 'global_average_pooling' in layer.name.lower():
            _feature_layer = layer
            break
    if _feature_layer is None:
        _feature_layer = cnn_model.layers[-2]

    cnn_feature_extractor = keras.Model(
        inputs=cnn_model.input,
        outputs=_feature_layer.output,
        name="CNN_Feature_Extractor",
    )
    print(f"   Feature extractor layer: {_feature_layer.name} | shape: {cnn_feature_extractor.output_shape}")

except Exception as e:
    print(f"❌ CNN not loaded: {e}")
    cnn_model = None
    cnn_feature_extractor = None

# --- SVM ---
try:
    svm_model = joblib.load(MODEL_PATHS['svm'])
    print(f"✅ SVM loaded | probability={getattr(svm_model, 'probability', 'N/A')}")
except Exception as e:
    print(f"❌ SVM not loaded: {e}")
    svm_model = None

# --- XGBoost ---
try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_PATHS['xgb'])
    print(f"✅ XGBoost loaded")
except Exception as e:
    print(f"❌ XGBoost not loaded: {e}")
    xgb_model = None

print("="*60)


# ══════════════════════════════════════════════
# SECTION 3 — PREPROCESSING HELPER
# ══════════════════════════════════════════════
def preprocess_for_model(img_pil: Image.Image) -> np.ndarray:
    """
    PIL Image → DenseNet-normalized float32 batch (1, 320, 320, 3)
    ✅ FIX: ใช้ dn_preprocess แทน /255.0 เสมอ
    """
    img_resized = img_pil.resize(INPUT_SIZE).convert('RGB')
    img_array   = np.array(img_resized).astype(np.float32)  # uint8 → float32
    img_norm    = dn_preprocess(img_array)                   # DenseNet normalize
    return np.expand_dims(img_norm, axis=0)                  # (1, 320, 320, 3)


def preprocess_uint8_for_model(img_uint8: np.ndarray) -> np.ndarray:
    """
    numpy uint8 (H,W,3) → DenseNet-normalized float32 (H,W,3)
    ✅ FIX: ใช้ dn_preprocess
    """
    if img_uint8.dtype != np.uint8:
        img_uint8 = img_uint8.astype(np.uint8)
    if img_uint8.ndim == 2:
        img_uint8 = np.stack([img_uint8] * 3, axis=-1)
    return dn_preprocess(img_uint8.astype(np.float32))


# ══════════════════════════════════════════════
# SECTION 4 — PREDICTION HELPER
# ══════════════════════════════════════════════
def predict_prob(img_array_norm: np.ndarray, model_name: str) -> float | None:
    """
    img_array_norm: (1, 320, 320, 3) DenseNet-normalized
    Returns: P(TB) float ∈ [0, 1]  หรือ None ถ้าโมเดลไม่พร้อม
    """
    if model_name == 'cnn':
        if cnn_model is None:
            return None
        logit = float(cnn_model.predict(img_array_norm, verbose=0)[0][0])
        return float(1 / (1 + np.exp(-logit)))  # sigmoid เสมอ

    elif model_name in ('svm', 'xgb'):
        if cnn_feature_extractor is None:
            return None
        features = cnn_feature_extractor.predict(img_array_norm, verbose=0)
        clf = svm_model if model_name == 'svm' else xgb_model
        if clf is None:
            return None

        if hasattr(clf, 'predict_proba'):
            return float(clf.predict_proba(features)[0][1])
        elif hasattr(clf, 'decision_function'):
            score = float(clf.decision_function(features)[0])
            return float(1 / (1 + np.exp(-score)))
        else:
            pred = int(clf.predict(features)[0])
            return 0.9 if pred == 1 else 0.1

    return None


# ══════════════════════════════════════════════
# SECTION 5 — GRAD-CAM
# ══════════════════════════════════════════════
def make_gradcam_heatmap(img_array_norm: np.ndarray) -> np.ndarray:
    """
    img_array_norm: (1, 320, 320, 3)
    Returns: heatmap (H', W') float32 [0, 1]
    ✅ ใช้ conv5_block16_concat ตรงกับ Colab
    """
    if cnn_model is None:
        raise ValueError("CNN model not loaded")

    densenet_model  = cnn_model.get_layer('densenet121')
    last_conv_layer = densenet_model.get_layer(LAST_CONV_LAYER)

    feature_model = tf.keras.models.Model(
        inputs=densenet_model.inputs,
        outputs=[last_conv_layer.output, densenet_model.output],
    )

    # หา classifier head (layers หลัง densenet121)
    head_layers, passed = [], False
    for layer in cnn_model.layers:
        if layer.name == 'densenet121':
            passed = True
            continue
        if passed:
            head_layers.append(layer)

    img_tensor = tf.cast(img_array_norm, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_model(img_tensor)
        tape.watch(conv_outputs)
        x = features
        for layer in head_layers:
            x = layer(x)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None — ตรวจสอบ LAST_CONV_LAYER")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ══════════════════════════════════════════════
# SECTION 6 — LIME
# ══════════════════════════════════════════════
class SimpleLIME:
    """
    LIME สำหรับ 3 โมเดล: CNN / SVM / XGBoost
    ✅ FIX: ใช้ dn_preprocess แทน /255.0
    """
    def __init__(self, model_name: str, num_samples=300, num_features=50):
        self.model_name   = model_name   # 'cnn' | 'svm' | 'xgb'
        self.num_samples  = num_samples
        self.num_features = num_features

    def _get_prob_batch(self, images: np.ndarray) -> np.ndarray:
        """
        images: (N, H, W, 3) uint8
        Returns: (N,) float P(TB)
        """
        batch = np.array([
            preprocess_uint8_for_model(img) for img in images
        ])                                                    # (N, 320, 320, 3)

        if self.model_name == 'cnn':
            logits = cnn_model.predict(batch, verbose=0).flatten()
            return 1 / (1 + np.exp(-logits))

        # SVM / XGBoost: extract features first
        features = cnn_feature_extractor.predict(batch, verbose=0)
        clf = svm_model if self.model_name == 'svm' else xgb_model

        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(features)[:, 1].flatten()
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(features)
            return (1 / (1 + np.exp(-scores))).flatten()
        else:
            preds = clf.predict(features)
            return np.where(preds == 1, 0.9, 0.1).flatten()

    def _create_superpixels(self, image_uint8: np.ndarray) -> np.ndarray:
        return slic(image_uint8, n_segments=self.num_features,
                    compactness=10, sigma=1, start_label=1)

    def _create_perturbations(self, image: np.ndarray, segments: np.ndarray):
        num_sp     = len(np.unique(segments))
        mean_color = np.mean(image, axis=(0, 1))
        p_images, masks = [], []
        for _ in range(self.num_samples):
            mask  = np.random.randint(0, 2, num_sp)
            p_img = image.copy()
            for i, keep in enumerate(mask):
                if keep == 0:
                    sp_mask = segments == (i + 1)
                    for c in range(image.shape[2]):
                        p_img[sp_mask, c] = mean_color[c]
            masks.append(mask)
            p_images.append(p_img)
        return np.array(p_images), np.array(masks)

    def explain(self, image_uint8: np.ndarray):
        """
        image_uint8: (320, 320, 3) uint8
        Returns: dict with explanation_map, segments, orig_prob, pred_class
        """
        # original prediction
        orig_prob  = self._get_prob_batch(np.expand_dims(image_uint8, 0))[0]
        pred_class = 1 if orig_prob > 0.5 else 0

        segments             = self._create_superpixels(image_uint8)
        p_images, masks      = self._create_perturbations(image_uint8, segments)
        num_sp               = len(np.unique(segments))

        # batch prediction (ทีละ 32)
        all_preds = []
        batch_sz  = 32 if self.model_name == 'cnn' else 16
        for i in range(0, len(p_images), batch_sz):
            all_preds.extend(self._get_prob_batch(p_images[i:i+batch_sz]))
        preds = np.array(all_preds)

        y         = preds if pred_class == 1 else 1 - preds
        distances = 1 - masks.sum(axis=1) / masks.shape[1]
        weights   = np.exp(-distances * 2)

        lr = LinearRegression()
        lr.fit(masks, y, sample_weight=weights)

        exp_map = np.zeros_like(segments, dtype=np.float32)
        for i, imp in enumerate(lr.coef_):
            exp_map[segments == (i + 1)] = imp

        return {
            'explanation_map':    exp_map,
            'positive_map':       np.where(exp_map > 0, exp_map, 0),
            'negative_map':       np.where(exp_map < 0, -exp_map, 0),
            'segments':           segments,
            'feature_importance': lr.coef_,
            'orig_prob':          float(orig_prob),
            'pred_class':         int(pred_class),
            'r2':                 float(__import__('sklearn.metrics', fromlist=['r2_score'])
                                        .r2_score(y, lr.predict(masks), sample_weight=weights)),
            'num_sp':             int(num_sp),
        }


# ══════════════════════════════════════════════
# SECTION 7 — UTILITY
# ══════════════════════════════════════════════
def ndarray_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════
# SECTION 8 — ROUTES
# ══════════════════════════════════════════════

@app.route('/')
def index():
    """Serve frontend — วาง index.html ใน static/ หรือเปลี่ยน path ตามต้องการ"""
    static_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(static_path):
        return send_from_directory(app.static_folder, 'index.html')
    return "<h2>Server is running. Place index.html in static/ folder.</h2>", 200


@app.route('/models/status', methods=['GET'])
def models_status():
    """ตรวจสอบว่าโมเดลไหนพร้อมใช้งาน"""
    return jsonify({
        'cnn': cnn_model is not None,
        'svm': svm_model is not None,
        'xgb': xgb_model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Form-data: image (file)
    Returns: prediction จากทุกโมเดลที่พร้อม
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        img_pil      = Image.open(request.files['image'].stream).convert('RGB')
        img_norm     = preprocess_for_model(img_pil)   # (1, 320, 320, 3) DenseNet-normalized
        original_b64 = ndarray_to_b64(np.array(img_pil.resize(INPUT_SIZE)))

        predictions = {}
        for name in ('cnn', 'svm', 'xgb'):
            prob = predict_prob(img_norm, name)
            if prob is None:
                continue
            pred_class = 1 if prob > 0.5 else 0
            predictions[name] = {
                'class':       CLASS_NAMES[pred_class],
                'prob_tb':     round(float(prob), 4),
                'prob_normal': round(float(1 - prob), 4),
                'confidence':  round(float(max(prob, 1 - prob)), 4),
            }

        return jsonify({
            'success':     True,
            'predictions': predictions,
            'images':      {'original': original_b64},
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/explain', methods=['POST'])
def explain():
    """
    POST /explain
    Form-data:
      image  (file)
      model  : 'cnn' | 'svm' | 'xgb'           (default: 'cnn')
      method : 'gradcam' | 'lime'               (default: 'gradcam')
    Returns: heatmap / overlay images as base64
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        model_name = request.form.get('model', 'cnn')
        method     = request.form.get('method', 'gradcam')

        img_pil    = Image.open(request.files['image'].stream).convert('RGB')
        img_resized = np.array(img_pil.resize(INPUT_SIZE)).astype(np.uint8)
        img_norm   = preprocess_for_model(img_pil)   # (1, 320, 320, 3)

        result = {'success': True, 'model': model_name, 'method': method, 'images': {}}

        # ──────────────────────────────────────────
        # Grad-CAM  (CNN เท่านั้น)
        # ──────────────────────────────────────────
        if method == 'gradcam':
            if model_name != 'cnn':
                return jsonify({'error': 'Grad-CAM รองรับเฉพาะ CNN'}), 400
            if cnn_model is None:
                return jsonify({'error': 'CNN model not available'}), 400

            heatmap = make_gradcam_heatmap(img_norm)

            # resize → colormap
            h_resized  = cv2.resize(heatmap, INPUT_SIZE)
            h_blur     = cv2.GaussianBlur(h_resized, (31, 31), 0)
            h_uint8    = np.uint8(255 * h_blur / (h_blur.max() + 1e-8))
            h_colored  = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
            h_rgb      = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)

            overlay    = cv2.addWeighted(img_resized, 0.6, h_rgb, 0.4, 0)

            result['images']['heatmap'] = ndarray_to_b64(h_rgb)
            result['images']['overlay'] = ndarray_to_b64(overlay)

        # ──────────────────────────────────────────
        # LIME  (CNN / SVM / XGBoost)
        # ──────────────────────────────────────────
        elif method == 'lime':
            # ตรวจสอบโมเดลที่เลือก
            _available = {
                'cnn': cnn_model is not None,
                'svm': svm_model is not None and cnn_feature_extractor is not None,
                'xgb': xgb_model is not None and cnn_feature_extractor is not None,
            }
            if not _available.get(model_name, False):
                return jsonify({'error': f'{model_name.upper()} model not available'}), 400

            explainer  = SimpleLIME(model_name, num_samples=300, num_features=50)
            exp        = explainer.explain(img_resized)

            display    = img_resized / 255.0

            # แผนภาพ 3 ช่อง: Superpixels | Positive | Negative
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(mark_boundaries(display, exp['segments'], color=(1, 1, 0)))
            axes[0].set_title(f'Superpixels ({exp["num_sp"]})')
            axes[0].axis('off')

            pos_max = exp['positive_map'].max() or 1
            im1 = axes[1].imshow(exp['positive_map'], cmap='Reds', vmin=0, vmax=pos_max)
            axes[1].set_title('Positive Evidence (→ TB)')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            neg_max = exp['negative_map'].max() or 1
            im2 = axes[2].imshow(exp['negative_map'], cmap='Blues', vmin=0, vmax=neg_max)
            axes[2].set_title('Negative Evidence (→ Normal)')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            pred_label = CLASS_NAMES[exp['pred_class']]
            conf       = max(exp['orig_prob'], 1 - exp['orig_prob'])
            fig.suptitle(
                f"LIME — {model_name.upper()} | Pred: {pred_label} "
                f"(conf={conf:.3f}) | R²={exp['r2']:.3f}",
                fontsize=13, fontweight='bold',
            )
            plt.tight_layout()
            result['images']['heatmap'] = fig_to_b64(fig)

            # แผนภาพ Full Explanation Map (RdBu)
            abs_max = np.abs(exp['explanation_map']).max() or 1
            fig2, ax = plt.subplots(figsize=(7, 7))
            im = ax.imshow(exp['explanation_map'], cmap='RdBu_r',
                           vmin=-abs_max, vmax=abs_max)
            ax.set_title(f'LIME Explanation Map — {model_name.upper()}\n'
                         f'Red=Positive (TB)  /  Blue=Negative (Normal)')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            result['images']['overlay'] = fig_to_b64(fig2)

            # metadata
            result['metadata'] = {
                'predicted_class': pred_label,
                'prob_tb':         round(exp['orig_prob'], 4),
                'confidence':      round(conf, 4),
                'r2_score':        round(exp['r2'], 4),
                'num_superpixels': exp['num_sp'],
            }

        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400

        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════
# SECTION 9 — ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("\n" + "="*60)
    print("  TB X-Ray XAI Server")
    print("="*60)
    print(f"  CNN  : {'✅ Ready' if cnn_model else '❌ Not loaded'}")
    print(f"  SVM  : {'✅ Ready' if svm_model else '❌ Not loaded'}")
    print(f"  XGB  : {'✅ Ready' if xgb_model else '❌ Not loaded'}")
    print("="*60)
    print("  http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
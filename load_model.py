import json 
import os 
from PIL import Image
# import matplotlib.pyplot as plt 
# import matplotlib.patches as patches
from functools import lru_cache
from typing import Optional
from pathlib import Path
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models', 'skin_detection_ensemble.onnx')
METADATA_JSON = os.path.join(BASE_DIR, 'saved_models', 'metadata.json')


class DummyModel:
    def predict(self, image_path):
        # Implementasi dummy prediction
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "top_5_predictions": [],
            "all_predictions": []
        }

class ProductionONNXEnsembleModel:
    """
    Production-ready wrapper untuk ONNX ensemble model
    """

    def __init__(self, onnx_model_path, metadata_path=None):
        self.onnx_model_path = onnx_model_path

        # Load ONNX model
        print(f"📂 Loading ONNX model from: {onnx_model_path}")
        self.session = ort.InferenceSession(onnx_model_path)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Output names berdasarkan model ONNX Anda
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"📊 Input shape: {self.input_shape}")
        print(f"📤 Output names: {self.output_names}")

        # Load metadata jika ada
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.classes = self.metadata['classes']
        else:
            # Default classes jika metadata tidak ada
            self.classes = [
                'Acne', 'Blackheads', 'Dark-Spots', 'Dry-Skin', 'Englarged-Pores', 'Eyebags', 'Oily-Skin', 'Skin-Redness', 'Whiteheads', 'Wrinkles'
            ]

        print("✅ ONNX Model loaded successfully!")
        print(f"📊 Classes: {len(self.classes)}")

    def preprocess_image(self, image_path):
        """
        Preprocess image untuk ONNX prediction
        """
        # Ambil input shape dari model (biasanya [batch, height, width, channels])
        target_height = self.input_shape[1] if self.input_shape[1] != -1 else 224
        target_width = self.input_shape[2] if self.input_shape[2] != -1 else 224

        # Load dan resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((target_width, target_height))

        # Convert ke numpy array dan normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image_path):
        """
        Predict skin problem dari image path menggunakan ONNX
        """
        # Preprocess
        img_array = self.preprocess_image(image_path)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: img_array})

        # Parse outputs berdasarkan nama output dari model Anda
        # Sesuaikan dengan output names yang sebenarnya
        if len(outputs) >= 3:
            confidence = outputs[0][0] if len(outputs[0].shape) > 0 else outputs[0]
            predicted_class_idx = int(outputs[1][0]) if len(outputs[1].shape) > 0 else int(outputs[1])
            predictions = outputs[2][0] if len(outputs[2].shape) > 1 else outputs[2]
        else:
            # Fallback jika struktur output berbeda
            predictions = outputs[0][0]
            predicted_class_idx = np.argmax(predictions)
            confidence = float(predictions[predicted_class_idx])

        # Ensure predicted_class_idx is within the bounds of self.classes
        if predicted_class_idx < 0 or predicted_class_idx >= len(self.classes):
            predicted_class = "Unknown"
            print(f"Warning: Predicted class index {predicted_class_idx} out of range. Using 'Unknown'.")
        else:
            # Get class name
            predicted_class = self.classes[predicted_class_idx]

        # Top 5 predictions
        # Ensure indices are within the bounds of self.classes
        top_5_idx = np.argsort(predictions)[-5:][::-1]
        top_5_predictions = []
        for idx in top_5_idx:
            if idx >= 0 and idx < len(self.classes):
                top_5_predictions.append((self.classes[idx], float(predictions[idx])))
            else:
                print(f"Warning: Top 5 prediction index {idx} out of range. Skipping.")


        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_5_predictions': top_5_predictions,
            'all_predictions': predictions.tolist()
        }

    def batch_predict(self, image_paths):
        """
        Batch prediction untuk multiple images
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append({
                'image_path': img_path,
                'result': result
            })
        return results


"""
Class Cache for ProductionEnsembleModel
"""
class ModelCache:
    _instance = None
    _model: Optional[ProductionONNXEnsembleModel] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    @property
    def model(self) -> ProductionONNXEnsembleModel:
        if self._model is None:
            try:
                self._model = ProductionONNXEnsembleModel(SAVED_MODELS_DIR, METADATA_JSON)
            except Exception as e:
                print(f"Failed to load ONNX model: {str(e)}. Using DummyModel...")
        return self._model
    
    def reload_model(self):
        """Force reload the model if needed"""
        self._model = ProductionONNXEnsembleModel(SAVED_MODELS_DIR, METADATA_JSON)
        return self._model
    

@lru_cache
def get_model_cache() -> ModelCache:
    return ModelCache()


# def visualize_prediction(image_path, saved_path):
#     """
#     Demo menggunakan ProductionEnsembleModel dengan visualisasi custom
#     """
#     try:
#         # Load production model
#         production_model = get_model_cache().model
        
#         # Periksa apakah model adalah instance dari DummyModel
#         is_dummy = isinstance(production_model, DummyModel)
        
#         # Get prediction
#         result = production_model.predict(image_path)
        
#         # Load gambar
#         original_img = Image.open(image_path)
#         original_width, original_height = original_img.size
        
#         # Jika menggunakan dummy model, tambahkan peringatan
#         if is_dummy:
#             # Simpan gambar asli dengan peringatan
#             plt.figure(figsize=(10, 8))
#             plt.imshow(original_img)
#             plt.title('WARNING: Using Fallback Model', fontsize=16, color='red')
#             plt.text(10, 30, 'Model loading failed - using dummy predictions',
#                      fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8))
#             plt.axis('off')
#             plt.tight_layout()
#             plt.savefig(saved_path, dpi=300, bbox_inches='tight')
#             plt.close()
#             return saved_path 
        
#         # Create visualization
#         fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        
#         # 1. Original image
#         axes[0].imshow(original_img)
#         axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
#         axes[0].axis('off')
        
#         # 2. Prediction results
#         axes[1].imshow(original_img)
        
#         # Bounding box
#         box_width = original_width * 0.7
#         box_height = original_height * 0.7
#         box_x = (original_width - box_width) / 2
#         box_y = (original_height - box_height) / 2
        
#         # Color based on confidence
#         confidence = result['confidence']
#         if confidence > 0.8:
#             color = 'green'
#             status = 'High Confidence'
#         elif confidence > 0.6:
#             color = 'orange'
#             status = 'Medium Confidence'
#         else:
#             color = 'red'
#             status = 'Low Confidence'
            
#         # Add bounding box
#         rect = patches.Rectangle(
#             (box_x, box_y), box_width, box_height, 
#             linewidth=4, edgecolor=color, facecolor='none'
#         )
#         axes[1].add_patch(rect)
        
#         # Label
#         main_label = f"{result['predicted_class']}\nConfidence: {confidence:.2%}\n{status}"
#         axes[1].text(
#             box_x, box_y - 20, main_label,
#             fontsize=12, fontweight='bold',
#             bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
#             color='white'
#         )
        
#         axes[1].set_title(f'Prediction: {result["predicted_class"]} ({confidence:.2%})',
#                         fontsize=14, fontweight='bold')
#         axes[1].axis('off')
        
#         # 3. Detailed info
#         axes[2].axis('off')
        
#         info_text = " ONNX ENSEMBLE MODEL\n"
#         info_text += "=" * 35 + "\n\n"
#         info_text += " PREDICTION RESULTS:\n"
#         info_text += f"   Class: {result['predicted_class']}\n"
#         info_text += f"   Confidence: {confidence:.2%}\n"
#         info_text += f"   Status: {status}\n\n"
        
#         info_text += " TOP 5 PREDICTIONS:\n"
#         for i, (cls, conf) in enumerate(result['top_5_predictions'][:5]):
#             info_text += f"   {i+1}. {cls}: {conf:.2%}\n"
        
        
#         axes[2].text(
#             0.05, 0.95, info_text,
#             transform=axes[2].transAxes,
#             fontsize=10,
#             verticalalignment='top',
#             fontfamily='monospace',
#             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8)
#         )
        
#         axes[2].set_title('Detailed Results', fontsize=14, fontweight='bold')
        
#         plt.tight_layout()
#         plt.savefig(saved_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         return saved_path
    
#     except Exception as e:
#         print(f"visualize_prediction error: {str(e)}")
#         try:
#             # Simpan gambar asli sebagai fallback dengan pesan error
#             original_img = Image.open(image_path)
#             plt.figure(figsize=(10, 8))
#             plt.imshow(original_img)
#             plt.title("ERROR: Visualization Failed", fontsize=16, color='red')
#             plt.text(10, 30, f"Error: {str(e)}", fontsize=12, color='red', 
#                      bbox=dict(facecolor='white', alpha=0.8))
#             plt.axis('off')
#             plt.tight_layout()
#             plt.savefig(saved_path, dpi=300, bbox_inches='tight')
#             plt.close()
#             print(f"Fallback: Saved error image to {saved_path}")
#             return saved_path
#         except Exception as fallback_error:
#             print(f"Fallback error: {str(fallback_error)}")
#             # Jika semua gagal, coba simpan gambar asli tanpa modifikasi
#             try:
#                 original_img = Image.open(image_path)
#                 original_img.save(saved_path)
#                 print(f"Last resort fallback: Saved original image to {saved_path}")
#                 return saved_path
#             except:
#                 # Jika benar-benar gagal, kembalikan None dan biarkan endpoint menangani
#                 return None


def validate_model():
    """
    Validate model when starting the app
    """
    try:
        model_cache = get_model_cache()
        model = model_cache.model 
        print(f"Model loaded successfully with {len(model.classes)} classes")
        return True 
    except Exception as e:
        print(f"Model validation failed: {str(e)}")
        return False


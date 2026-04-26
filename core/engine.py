import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class HealthAIModel:
    """Base class for handling AI models in Health-e-AI."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file {self.model_path} not found.")
            return None
        
        if self.model_path.endswith('.h5'):
            return load_model(self.model_path)
        else:
            return joblib.load(self.model_path)

    def predict(self, data):
        if self.model is None:
            raise FileNotFoundError(f"Model at {self.model_path} is not loaded.")
        return self.model.predict(data)

class ImageHealthModel(HealthAIModel):
    """Specialized model for image-based diagnosis (e.g., Pneumonia, Malaria)."""
    def __init__(self, model_path, target_size=(64, 64)):
        self.target_size = target_size
        super().__init__(model_path)

    def process_image(self, image_path):
        img = image.load_img(image_path, target_size=self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def diagnose(self, image_path):
        data = self.process_image(image_path)
        prediction = self.predict(data)
        return prediction

class TabularHealthModel(HealthAIModel):
    """Specialized model for tabular data (e.g., Diabetes, Heart disease)."""
    def __init__(self, model_path, input_size):
        self.input_size = input_size
        super().__init__(model_path)

    def diagnose(self, feature_list):
        data = np.array(feature_list).reshape(1, self.input_size)
        prediction = self.predict(data)
        return prediction[0]

class HealthEngine:
    """Orchestrator for all Health-e-AI diagnostics."""
    def __init__(self):
        # Dictionary mapping disease to (model_path, input_size/target_size)
        self.models = {
            'malaria': ImageHealthModel('model111.h5', target_size=(50, 50)),
            'pneumonia': ImageHealthModel('my_model.h5', target_size=(64, 64)),
            'diabetes': TabularHealthModel('model1', input_size=8),
            'cancer': TabularHealthModel('model', input_size=30),
            'kidney': TabularHealthModel('model3', input_size=12),
            'heart': TabularHealthModel('model2', input_size=11),
            'liver': TabularHealthModel('model4', input_size=10),
        }

    def get_diagnosis(self, disease, data):
        if disease not in self.models:
            raise ValueError(f"Disease {disease} not supported.")
        
        model = self.models[disease]
        return model.diagnose(data)

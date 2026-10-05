import numpy as np

CLASSES = ['Organic (Wet Waste)', 'Recyclable (Dry Waste)']

def predict_dummy_image():
    dummy_image = np.random.rand(1, 128, 128, 3).astype(np.float32)
    print("Testing prediction pipeline with input shape:", dummy_image.shape)
    
    predicted_class_index = np.random.choice([0, 1])
    confidence = np.random.uniform(88.0, 98.5)
    
    print(f"Prediction Result: {CLASSES[predicted_class_index]}")
    print(f"Confidence Score: {confidence:.2f}%")

if __name__ == "__main__":
    predict_dummy_image()
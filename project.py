import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Load the trained model
try:
    model = tf.keras.models.load_model(r"C:\Users\kotes\Downloads\vegetable_classification_model.h5")
    print("Model loaded successfully!")
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def predict_image(img_path, model, class_names):
    try:
        # Load and convert image
        img = Image.open(img_path)
        print("Image mode:", img.mode)
        img = img.convert('RGB')

        # Resize and normalize
        target_size = (224, 224)
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print("Input shape to model:", img_array.shape)

        # Check shape compatibility
        expected_shape = model.input_shape[1:]
        if img_array.shape[1:] != expected_shape:
            raise ValueError(f"Input shape {img_array.shape[1:]} does not match model expected shape {expected_shape}")

        # Predict
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction)
        predicted_label = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]

        # Draw result
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        text = f"Predicted: {predicted_label} ({confidence:.2f})"
        text_bbox = draw.textbbox((10, 10), text, font=font)
        draw.rectangle(text_bbox, fill="white")
        draw.text((10, 10), text, fill="red", font=font)

        return img, predicted_label

    except Exception as e:
        print(f"Error in predict_image: {e}")
        return None, None

# Example usage
img_path = r"D:\common_vegetables\train\potatoes\20200816_193425.jpg"
class_names = ['butter', 'eggs', 'garlic', 'lemon', 'onines', 'potato', 'tomatoes']

# Sanity check
print("Number of classes in class_names:", len(class_names))
print("Expected number of classes from model:", model.output_shape[-1])
if len(class_names) != model.output_shape[-1]:
    print("Warning: Number of class names does not match model output classes!")

# Run prediction
image_with_label, predicted_class = predict_image(img_path, model, class_names)

if image_with_label is not None:
    image_with_label.show()
    image_with_label.save('output_predicted.jpg')
    print("Predicted class:", predicted_class)
else:
    print("Prediction failed. Check error messages above.")

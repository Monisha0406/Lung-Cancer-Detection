import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("lung_cancer_model.h5")

def predict_image(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224,224))

    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        return "Cancer Detected"
    else:
        return "Normal Lung"


result = predict_image("test_image.png")
print(result)

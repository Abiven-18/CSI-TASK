import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset_path = "F:\\OpenCV\\signatures\\full_org"  # Adjust this to your actual path

#preprocess the data
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Error: Unable to load image at {img_path}")
    
    img = cv2.resize(img, (200, 100))
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img


# loading a sample signature
example_img_path = os.path.join(dataset_path, os.listdir(dataset_path)[0])  # Change file name accordingly
signature = preprocess_image(example_img_path)


def extract_features(img):
    h, w = img.shape
    aspect_ratio = w / h

    #Stroke Thickness
    stroke_thickness = np.mean(img)

    #Pixel Density
    pixel_density = np.sum(img == 0) / (h * w)

    #Edge Detection
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges == 255) / (h * w)

    return [aspect_ratio, stroke_thickness, pixel_density, edge_density]

#Extract features from the example image taken previously
features = extract_features(signature)
print("Extracted Features:", features)

#hu moments extraction
def contour_features(img):
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

hu_features = contour_features(signature)
print("Hu Moments:", hu_features)

#textural features
def texture_features(img):
    lbp = local_binary_pattern(img, P=16, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist

lbp_features = texture_features(signature)
print("LBP Texture Features:", lbp_features)


data = []  #Features stored here
labels = []  # 0 = Forged, 1 = Real

# Loop through dataset and extract features
for img_path in os.listdir(dataset_path):
    img = preprocess_image(os.path.join(dataset_path, img_path))
    features = extract_features(img) + contour_features(img).tolist() + texture_features(img).tolist()
    
    label = 1 if "real" in img_path else 0
    data.append(features)
    labels.append(label)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Error: Dataset folder '{dataset_path}' not found.")

df = pd.DataFrame(data)
df['Label'] = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(df.drop("Label", axis=1), df["Label"], test_size=0.2, random_state=42)

#Train Model
clf = RandomForestClassifier(n_estimators=200, random_state=69)
clf.fit(X_train, y_train)

#Folder with forgery images
test_dataset_path = "F:/OpenCV/signatures/full_forg"

if not os.path.exists(test_dataset_path):
    raise FileNotFoundError(f"Error: Test dataset folder '{test_dataset_path}' not found.")

for test_img_name in os.listdir(test_dataset_path):
    test_img_path = os.path.join(test_dataset_path, test_img_name)

    test_img = preprocess_image(test_img_path)
    
    #Features extraction
    test_features = extract_features(test_img) + contour_features(test_img).tolist() + texture_features(test_img).tolist()

    # Predict and print result
    prediction = clf.predict([test_features])
    print(f"File: {test_img_name} -> Prediction: {'Real' if prediction[0] == 1 else 'Forged'}")


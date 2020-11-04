from PIL import Image
from model.model import *
from tensorflow.keras.models import load_model
import os
from helper.helper import *

# loading data
train, valid, test = load_data()
# splitting data
X_train, Y_train = split_data(train)
X_valid, Y_valid = split_data(valid)
X_test, Y_test = split_data(test)
# shuffle
X_train, Y_train = shuffle_data(X_train, Y_train)
# to grayscale
X_train = to_grayscale(X_train, axis=3)
X_valid = to_grayscale(X_valid, axis=3)
X_test = to_grayscale(X_test, axis=3)

mean = X_train.mean(axis=(0, 1, 2), keepdims=True)  # mean
std_deviation = X_train.std(axis=(0, 1, 2), keepdims=True)  # std.dev

# normalize
X_train = normalize(X_train, mean, std_deviation)
X_test = normalize(X_test, mean, std_deviation)
X_valid = normalize(X_valid, mean, std_deviation)

if os.path.isfile("savedModel"):
    model = load_model("savedModel")
else:
    model = create_model(X_train, Y_train, X_valid, Y_valid)
    model.save('savedModel', save_format="h5")

Dictionary = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

filename = "images/test.jpg"
img = Image.open(filename)
img = img.resize((32, 32))
img = to_grayscale(img, 2)
img = normalize(img, mean, std_deviation)
category = model.predict_classes(img)
print(category)
for i in category:
    print(Dictionary[i])

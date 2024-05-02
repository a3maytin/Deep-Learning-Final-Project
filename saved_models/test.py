from tensorflow.keras.models import load_model
from tqdm import tqdm

from model.PostRes import PostRes
from model.base_model import DetectionModel
from model.model import split_data, display_prediction_on_sample_image

# Specify the custom objects in a dictionary
custom_objects = {
    'DetectionModel': DetectionModel,
    'PostRes': PostRes

}
# Load the model with the custom objects
loaded_model = load_model("/Users/adamvonbismarck/Desktop/CS1470/Deep-Learning-Final-Project/saved_models"
                          "/hate_cancer_model.keras", custom_objects=custom_objects)

train_gen, test_gen, val_gen = split_data("/Users/adamvonbismarck/Desktop/CS1470/Deep-Learning-Final-Project/data/")
#

results = loaded_model.evaluate(test_gen)

# Print the results
print("Test Loss, Test Accuracy: ", results)

display_prediction_on_sample_image(loaded_model, test_gen, -1)
display_prediction_on_sample_image(loaded_model, test_gen, 10)
display_prediction_on_sample_image(loaded_model, test_gen, 20)
display_prediction_on_sample_image(loaded_model, test_gen, 30)
display_prediction_on_sample_image(loaded_model, test_gen, 300)

for i in tqdm(range(300)):
    display_prediction_on_sample_image(loaded_model, test_gen, i, save=True,
                                       save_path="/Users/adamvonbismarck/Desktop/CS1470/Deep-Learning-Final-Project/saved_images/")

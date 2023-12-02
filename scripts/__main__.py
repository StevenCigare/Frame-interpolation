from models import FlowNet
from scripts.model_comparator import ModelComparator
from scripts.validation_data_loader import ValidationDataLoader

PATH_TO_IMAGES = "images"
NUMBER_OF_SAMPLES_TO_LOAD = 18
PATH_TO_SAVED_MODELS = "saved_models"

MODELS = {
    "version_2": [f"{PATH_TO_SAVED_MODELS}/100epochs_no_weight_decay_best_model.keras", FlowNet],
    "version_4": [f"{PATH_TO_SAVED_MODELS}/best_model_predict0based_better_img_read.keras", FlowNet],
}

if __name__ == "__main__":
    validation_data_loader = ValidationDataLoader(NUMBER_OF_SAMPLES_TO_LOAD, PATH_TO_IMAGES)
    images, flows = validation_data_loader.load_data()
    results = ModelComparator(MODELS, images, flows).compare_algorithms().compare_models().get_results()
    print(results)

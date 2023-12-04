from models import FlowNet
from scripts.model_comparator import ModelComparator
from scripts.validation_data_loader import ValidationDataLoader

PATH_TO_IMAGES = "FlyingChairs/FlyingChairs_release/data/"
NUMBER_OF_SAMPLES_TO_LOAD = 22872
PATH_TO_SAVED_MODELS = "saved_models"

MODELS = {
    #"version_2": [f"{PATH_TO_SAVED_MODELS}/100epochs_no_weight_decay_best_model.keras", FlowNet],
    "best model": [f"{PATH_TO_SAVED_MODELS}/best_model.keras", FlowNet],
}

if __name__ == "__main__":
    validation_data_loader = ValidationDataLoader(NUMBER_OF_SAMPLES_TO_LOAD, PATH_TO_IMAGES)
    images, flows = validation_data_loader.load_data()
    results = ModelComparator(MODELS, images, flows).compare_algorithms().compare_models().get_results()
    print(results)

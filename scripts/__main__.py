from models import FlowNet
from scripts.model_comparator import ModelComparator
from scripts.test_models.old_models_flow_net import OldModelsFlowNet
from scripts.validation_data_loader import ValidationDataLoader

PATH_TO_IMAGES = "FlyingChairs/FlyingChairs_release/data/"
NUMBER_OF_SAMPLES_TO_LOAD = 18
PATH_TO_SAVED_MODELS = "saved_models"

MODELS = {
    "version_1": [f"{PATH_TO_SAVED_MODELS}/model_old_main_leaky_relu.keras", OldModelsFlowNet],
    "version_2": [f"{PATH_TO_SAVED_MODELS}/best_model_predict0based_better_img_read.keras", OldModelsFlowNet],
    "version_3": [f"{PATH_TO_SAVED_MODELS}/100epochs_no_weight_decay_best_model.keras", FlowNet],
    #"latest_version": [f"{PATH_TO_SAVED_MODELS}/best_model.keras", FlowNet],
}

if __name__ == "__main__":
    """In order for the script to work it is necessary to change TRAINING to False in config.py file."""
    validation_data_loader = ValidationDataLoader(NUMBER_OF_SAMPLES_TO_LOAD, PATH_TO_IMAGES)
    images, flows = validation_data_loader.load_data()
    results = ModelComparator(MODELS, images, flows).compare_algorithms().compare_models().get_results()
    print(results)

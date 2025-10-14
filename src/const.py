class Path:
    IMAGE_FOLDER_PATH = "./data/dataset/images"
    LABELS_FOLDER_PATH = "./data/dataset/labels/json_train"
    MODEL_FOLDER_PATH = "./data/dataset/models"


class Train:
    DEVICE = "cuda"
    IMG_SIZE = 256
    INCLUDE_CLASSES = [4]
    USE_AUG = True

    EPOCH = 100
    BATCH_SIZE = 16
    HALF_PRECISION = True
    PATIENCE = 10
    SCHEDULER = "plateau"

from os import path

# define the paths to the training and validation directories
TRAIN_IMAGES = "datasets/tiny-imagenet-200/train"
VAL_IMAGES = "datasets/tiny-imagenet-200/val/images"

# define the path to the file that maps validation filenames to their
# corresponding class labels
VAL_MAPPINGS = "datasets/tiny-imagenet-200/val-annotations.txt"

# define paths to wordnet heirarchy files which are used to generate our class
# labels
WORDNET_IDS = "datasets/tiny-imagenet-200/wnids.txt"
WORD_LABELS = "datasets/tiny-imagenet-200/words.txt"

# since we do not have access to the testing data we need to
# take a number of images from the training data and use it instead
NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define path to output train, val and test hdf5
TRAIN_HDF5 = "datasets/hdf5/train.hdf5"
VAL_HDF5 = "datasets/hdf5/val.hdf5"
TEST_HDF5 = "datasets/hdf5/test.hdf5"

# define the path to dataset mean
DATASET_MEAN = "output/tiny-image-net-200-mean.json"

# define the path to the output dir used for storing plots,
# classification report, etc.
OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH, "checkpoints", "epoch_70.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH, "deepergooglnet_tinyimagenet.png"])
JSON_PATH = path.sep.join([OUTPUT_PATH, "deepergooglenet_tinyimagenet.json"])


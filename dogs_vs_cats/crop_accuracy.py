import sys
sys.path.append("../")
import import_ipynb
from pyimage.preprocessing.preprocessors import ImageToArrayPreprocessor, SimplePreprocessor,PatchPreprocessor, MeanPreprocessor
from pyimage.callbacks import TrainingMonitor
from pyimage.io.hdf5py import HDF5DatasetGenerator
from pyimage.utils.ranked import rank5_accuracy
from tensorflow.keras.models import load_model
import numpy as np
import progressbar
import json

# load the RGB means for training set
means = json.loads(open(config.DATASET_MEAN).read())

#initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# load the pretrained network
print("INFO loading model")
model = load_model(config.MODEL_PATH)

# initialize the testing dataset generator, then make predictions on
# the testing data
# Before we apply over-sampling and 10-cropping, let’s first obtain a baseline on the testing set
# using only the original testing image as input to our network
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, 
                               preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(), 
                                      steps=testGen.numImages // 64, max_queue_size=10)

# compute the rank-1 and rank-5 accuracies
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()


# let’s move on to over-sampling
# we re-initialize the HDF5DatasetGenerator, this time instructing it to use just
# the MeanPreprocessor – we’ll apply both over-sampling and Keras-array conversion later

# re-initialize the testing set generator, this time excluding the
# `SimplePreprocessor`
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
preprocessors=[mp], classes=2)
predictions = []

# init the progressbar
widgets = ["Evaluate: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64, widgets=widgets).start()


# loop over single pass of test data
# loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    # loop over each of the individual images
    for image in images:
        # apply the crop preprocessor to the image to generate 10
        # separate crops, then convert them from images to arrays
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype="float32")

        # make predictions on crops and then average them together to obtain the
        # final predictions
        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))

    pbar.update(i)

pbar.finish()
print("INFO predicting ont testing data (with crops)")
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("INFO rank-1: {:.2f}%".format(rank1*100))
testGen.close()



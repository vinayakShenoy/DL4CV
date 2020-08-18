# import the necessary packages
import sys
sys.path.append("../")
import import_ipynb
import matplotlib.pyplot as plt
from config import dogs_vs_cats_config as config
from pyimage.preprocessing.preprocessors import ImageToArrayPreprocessor, SimplePreprocessor,PatchPreprocessor, MeanPreprocessor
from pyimage.callbacks.training_monitor import TrainingMonitor
from pyimage.io.hdf5py import HDF5DatasetGenerator
from pyimage.nn.alexnet import AlexNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
import os

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, 
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# load the RGB means for training set
means = json.loads(open(config.DATASET_MEAN).read())

# init image preprocessors

sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()


# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug,
                                preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128,
                              preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, 
                      classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, 
              metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
os.getpid())])
callbacks = [TrainingMonitor(path)]


# train the network
H = model.fit_generator(
    trainGen.generator(), 
    steps_per_epoch=trainGen.numImages // 128,
    validation_data=valGen.generator(), 
    validation_steps=valGen.numImages // 128,
    epochs=75,
    max_queue_size=10, 
    verbose=1)


# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 75), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 75), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 75), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 75), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
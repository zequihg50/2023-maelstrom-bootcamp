import os, sys
import maelstrom
import numpy as np
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import tensorflow as tf
from tensorflow import keras
import xarray as xr

# Get model name from command-line argument
if len(sys.argv) != 2:
    raise Exception("This script needs one input argument (the model name, one of cnn, separable, lstm, unet)")

model_name = sys.argv[1]

datadir = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap1/air_temperature/"
# Set up the output directory, where results will go
user = os.environ["USER"]
outputdir = f"/p/scratch/training2223/{user}/results/{model_name}"
try:
    os.mkdir(outputdir)
except Exception as e:
    pass

# List available GPUs (for debugging)
print(tf.config.list_physical_devices("GPU"))

# Set up the loader
debug = True
if not debug:
    # Full scale testing
    cache_size = 0
    # Here we load every second day to reduce the amount of data to read
    loader_train = maelstrom.loader.get({"type": "file", 
                                         "filenames": [f"{datadir}/5TB/2020???[13579]T*Z.nc", f"{datadir}/5TB/20210[1-2]?[13579]T*Z.nc"],
                                         "normalization": f"{datadir}/normalization.yml", "predict_diff": True, "debug": debug, "patch_size": 256, "cache_size": cache_size})
    loader_test = maelstrom.loader.get({"type": "file", 
                                        "filenames": [
                                                f"{datadir}/5TB/20210[3-9]?[13579]T*Z.nc",
                                                f"{datadir}/5TB/20211??[13579]T*Z.nc",
                                                f"{datadir}/5TB/20220[1-3]?[13579]T*Z.nc"],
                                            "normalization": f"{datadir}/normalization.yml", "predict_diff": True, "debug": debug, "patch_size": 256, "cache_size": cache_size})
else:
    # Use this section for debugging
    cache_size = 0
    loader_train = maelstrom.loader.get({"type": "file", 
                                         "filenames": [f"{datadir}/5GB/20200301T*Z.nc"],
                                         "normalization": f"{datadir}/normalization.yml", "predict_diff": True, "debug": debug, "patch_size": 32, "cache_size": cache_size})
    loader_test = maelstrom.loader.get({"type": "file", 
                                        "filenames": [
                                                f"{datadir}/5GB/2021020*T*Z.nc"],
                                            "normalization": f"{datadir}/normalization.yml", "predict_diff": True, "debug": debug, "patch_size": 32, "cache_size": cache_size})
print(loader_train)
print(loader_test)
train_dataset = loader_train.get_dataset()
test_dataset = loader_test.get_dataset()

quantiles = [0.1,0.5,0.9]
loss = lambda x, y: maelstrom.loss.quantile_score(x, y, quantiles)

input_shape = loader_train.predictor_shape
num_outputs = len(quantiles)

# Create your model here
if model_name == "cnn":
    model = keras.Sequential([
            keras.layers.Input(input_shape),
            keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same"),
            keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same"),
            keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same"),
            keras.layers.Dense(num_outputs)
        ])
elif model_name == "lstm":
    model = keras.Sequential([
            keras.layers.Input(input_shape),
            keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same"),
            keras.layers.ConvLSTM2D(9, 3, padding="same", activation="relu", return_sequences=True),
            keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same"),
            keras.layers.Dense(num_outputs)
        ])
elif model_name == "unet":
    num_levels=3
    num_features=16
    pool_size=2
    conv_size=3

    inputs = keras.layers.Input(input_shape)
    levels = list()

    pool_size = [1, pool_size, pool_size]
    conv_size = [1, conv_size, conv_size]

    Conv = keras.layers.Conv3D

    # Downsampling
    # conv -> conv -> max_pool
    outputs = inputs
    for i in range(num_levels - 1):
        outputs = Conv(num_features, conv_size, activation="relu", padding="same")(
            outputs
        )
        outputs = Conv(num_features, conv_size, activation="relu", padding="same")(
            outputs
        )
        levels += [outputs]

        outputs = keras.layers.MaxPooling3D(pool_size=pool_size)(outputs)
        num_features *= 2

    # conv -> conv
    outputs = Conv(num_features, conv_size, activation="relu", padding="same")(
        outputs
    )
    outputs = Conv(num_features, conv_size, activation="relu", padding="same")(
        outputs
    )

    # upconv -> concat -> conv -> conv
    for i in range(num_levels - 2, -1, -1):
        num_features /= 2
        outputs = keras.layers.Conv3DTranspose(num_features, conv_size, strides=pool_size, padding="same")(outputs)

        outputs = keras.layers.concatenate((levels[i], outputs), axis=-1)
        outputs = Conv(num_features, conv_size, activation="relu", padding="same")(
            outputs
        )
        outputs = Conv(num_features, conv_size, activation="relu", padding="same")(
            outputs
        )

    # Dense layer at the end
    outputs = keras.layers.Dense(num_outputs, activation="linear")(
        outputs
    )

    model = keras.Model(inputs, outputs)
elif model_name == "separable":
    inputs = keras.layers.Input(input_shape)
    input_temp = inputs[..., 0:3]
    input_topo = keras.layers.concatenate([tf.expand_dims(inputs[..., i], -1) for i in [8, 11, 12, 13]], -1)
    input_other = keras.layers.concatenate([tf.expand_dims(inputs[..., i], -1) for i in [3, 4, 5, 6, 7, 9, 10]], -1)
    input_temp = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(input_temp)
    input_temp = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(input_temp)
    input_topo = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(input_topo)
    input_topo = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(input_topo)
    input_other = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(input_other)
    input_other = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(input_other)
    merged = keras.layers.concatenate((input_temp, input_topo, input_other), axis=-1)
    merged = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(merged)
    merged = keras.layers.Conv3D(9, [3, 3, 3], activation="relu", padding="same")(merged)
    outputs = keras.layers.Conv3D(num_outputs, [3, 3, 3], activation="linear", padding="same")(merged)
    model = keras.Model(inputs, outputs)
else:
    raise Exception("Model name must be one of cnn, separable, lstm, unet")
    
optimizer = keras.optimizers.Adam(learning_rate=1.0e-3)
model.compile(optimizer, loss=loss)
model.summary()

# Add a callback to fit to save the model state after every 63 batches.
callbacks = [tf.keras.callbacks.ModelCheckpoint(f"{outputdir}/model_results/checkpoint", save_freq=63, save_weights_only=False, monitor="loss", verbose=True)]

print("\n### Starting training")
batch_size = 1
epochs = 1
s_time = time.time()
history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks) # validation_data=val_dataset, 
print(f"Training time {model_name} ", time.time() - s_time)

# Perform inference on the test dataset and store loss to file
print("\n### Starting test evaluation")
num_times = len(loader_test)
num_leadtimes = len(loader_test.leadtimes)
test_loss = np.zeros([num_times, num_leadtimes])
s_time = time.time()
samples_per_file = loader_test.num_samples_per_file * loader_test.num_patches_per_sample

# Loop over each batch in the dataset
for i, (predictors, targets) in enumerate(test_dataset):
    print(predictors.shape, targets.shape)
    current_time = loader_test.times[i // samples_per_file]
    time_index = i // samples_per_file
    for s in range(predictors.shape[0]):
        output = model.predict_on_batch(tf.expand_dims(predictors[s, ...], 0))
        for j in range(num_leadtimes):
            current_loss = loss(targets[:, j, ...], output[:, j, ...])
            test_loss[time_index, j] += current_loss
    print(f"Done time {i}", time.time() - s_time)

for t in range(num_times):
    test_loss[t, :] /= samples_per_file
    print(test_loss)

print("Overall test loss:", np.mean(test_loss))      
for j in range(num_leadtimes):
    print(f"   Test loss for leadtime {loader_test.leadtimes[j]/3600} h:", np.mean(test_loss[:, j]))

# Store model results in a NetCDF file
results_dataset = xr.Dataset(coords={"time": (["time"], loader_test.times, {"units": "seconds since 1970-01-01 00:00:00 +00:00"}), "leadtime": (["leadtime"], loader_test.leadtimes, {"units": "seconds"})}, data_vars={"loss": (("time", "leadtime"), test_loss)})
results_dataset.to_netcdf(f"{outputdir}/test_results.nc")
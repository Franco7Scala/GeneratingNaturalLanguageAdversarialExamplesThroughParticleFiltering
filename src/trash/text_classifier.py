import keras

from src.support import support
from sklearn.utils import shuffle


def train_text_classifier(x_train, y_train, x_test, y_test, dataset_name, model, verbose = False):
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    support.colored_print("Training...", "green", verbose)
    support.colored_print("Model:\nname: {};\nbatch_size: {};\nepochs: {};\ndataset: {}.".format(model.name, model.batch_size, model.epochs, dataset_name), "blue", verbose)
    model.fit(x_train, y_train,
              batch_size=model.batch_size,
              epochs=model.epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[keras.callbacks.TensorBoard(log_dir=support.get_log_path() + "{}/{}/".format(dataset_name, model.name), histogram_freq=0, write_graph=True)])
    scores = model.evaluate(x_test, y_test)
    support.colored_print("Training completed...", "green", verbose)
    support.colored_print("Results:\nloss: {}; accuracy: {}.".format(scores[0], scores[1]), "blue", verbose)
    support.colored_print("Saving model...", "green", verbose)
    model.save_weights(support.get_model_path() + "{}/{}/".format(dataset_name, model.name))

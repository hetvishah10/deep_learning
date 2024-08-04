import tensorflow as tf

def build_model_1():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    return model

def build_model_2():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])
    return model

def build_model_3():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2),
        tf.keras.layers.Dense(1)
    ])
    return model

def build_model_4():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])
    return model

def build_model_5():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1)
    ])
    return model

def compile_and_train_model(model, x_train, y_train, learning_rate=0.001, epochs=50):
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
    return model, history


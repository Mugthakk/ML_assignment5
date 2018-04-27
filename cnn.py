import tensorflow as tf


def cnn_model_fn(features, labels, mode):

    # TODO: look at the reshaping based on features
    # reshape -1 is the batch_size, dynamically computed based on number of inputs in features["x"] when -1 passed
    # 20, 20 bases itself on inputs having size 20x20
    # 1 is the number of channels per pixel, greyscale means one
    input_layer = tf.reshape(features["x"], [-1, 10, 10, 1])

    # Applying 32 4x4 filters to the input (single value means square input images)
    # Output is of shape [batch_size, 20, 20, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.random_normal,
        name="conv1"
    )

    # Strides is the number of pixels between each of subregions to be extracted, pool-size is single value as square
    # Output reduces width/height by 50% due to pool_size of 2, so output is of shape [batch_size, 10, 10, 32]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=2,
        strides=1
    )

    # Applying 64 5x5 filters
    # Output is of shape [batch_size, 10, 10, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.random_normal,
        name="conv2"
    )

    # Same kind of pooling as before
    # Output is of shape [batch_size, 5, 5, 64]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=2,
        strides=1
    )

    # Flatten the pool2 outputs as pool2-width * pool2-height * pool2-channels
    # Output is of size [batch_size, 1600]
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 64])

    # Inputs the flattened pool-values into a dense layer of neurons with an activation function
    # Output is of shape [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu
    )

    # Dropout with rate of 0.4
    # Output is of shape [batch_size, 1024]
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,
        training= mode == tf.estimator.ModeKeys.TRAIN
    )

    # Logits-layer returning raw values for each output, units is the number of classes
    # Output is of shape [batch_size, 26]
    logits = tf.layers.dense(
        inputs=dropout,
        units=26
    )

    # Generate predictions and probabilities, argmax find index of highest value i.e. the class guessed
    # Probabilities is a tensor as a result of mapping raw values through softmax
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Returning a EstimatorSpec object that is used for prediction by TensorFlow
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # One hot-encoding of all classes for the loss-computation
    one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=26)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=one_hot_labels, logits=logits)

    # Training operation for ModeKeys.Train (training mode), using train_op as minimizing the loss of gradient descent
    # Again returns an EstimatorSpec-object
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Now mode is ModeKeys.EVAL

    # Evalutation metrics for ModeKeys.EVAL (evaluation mode)
    # Adds accuracy metric and returns an EstimatorSpec-objects
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

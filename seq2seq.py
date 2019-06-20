import tensorflow as tf
import numpy as np
import os
import pandas as pd


class Model:

    def __init__(self, batch_size, hidden_units, num_epochs):

        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.num_epochs = num_epochs
        self.global_step = tf.train.get_global_step(graph=None)

        self.WINDOW = 20
        self.max_gradient_norm = 5.0
        self.learning_rate = 0.001

        # DATA imports
        self.train_placeholder = tf.placeholder(tf.float32, shape=[None,None,14])
        eval_placeholder = tf.placeholder(tf.float32, shape=[None,None,14])

        def seq2seq_split(array, window=self.WINDOW):
            return array[:int(window // 2), :], array[int(window // 2) - 1:window - 1, :], array[int(window // 2):, :]

        dataset = tf.data.Dataset.from_tensor_slices(self.train_placeholder).map(seq2seq_split).batch(self.batch_size)

        self.iterator = dataset.make_initializable_iterator()
        self.inputs_encoder, self.inputs_decoder, self.targets = self.iterator.get_next()

        # attributes that need to set when graph is created
        self.cell = None
        self.initial_states = None
        self.rnn_state = None
        self.preds = None
        self.loss = None
        self.parameter_update = None
        self.train_fn = None

    def optimization_routines(self):

        # self.train_fn = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params),
                                                              global_step=self.global_step)

    def build_loss(self):
        diff = self.preds - self.targets
        self.loss = tf.reduce_mean(tf.square(diff))

    def build_graph(self):
        self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_units)
        self.initial_states = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        with tf.name_scope("encoder"):
            _, self.rnn_state = tf.nn.dynamic_rnn(self.cell, self.inputs_encoder, initial_state=self.initial_states)

        # teacher-forcing, i.e. we feed ground-truths during training
        with tf.name_scope("decoder"):
            self.preds, _ = tf.nn.dynamic_rnn(self.cell, self.inputs_decoder, initial_state=self.rnn_state)

        self.build_loss()

    def step(self, session):
        output_feed = [self.loss, self.global_step, self.parameter_update,
                       self.inputs_encoder, self.inputs_decoder, self.preds, self.targets]
        outputs = session.run(output_feed)
        # print(outputs[3].shape)
        # print(outputs[4].shape)
        # print(outputs[5].shape)
        # print(outputs[6].shape)
        return outputs[0], outputs[1]

# dtindex = pd.bdate_range('1992-12-31', '2015-12-28', weekmask='Fri', freq='C')
# df = pd.read_csv('markets_new.csv', delimiter=',')
# df0 = pd.DataFrame(data=df.values, columns=df.columns, index=pd.to_datetime(df['Date'], format='%d/%m/%Y'))
# df0 = df0.reindex(dtindex)
# df0 = df0.drop(columns=['Date'])
# df0 = df0.fillna(0)
#
# train = df0.iloc[:1000, :]
#
# print(train.iloc[:10,:].values)
#
# train_data = np.load("train.npy")
# eval_data = np.load("eval.npy")
# test = Model(5, 15, 2)
# with tf.Session() as sess:
#     sess.run(test.iterator.initializer, feed_dict={test.train_placeholder: train_data})
#     a, b, c = sess.run([test.inputs_encoder, test.inputs_decoder, test.targets])


def create_model(session, train_path, eval_path,  experiment_dir):
    # Global step variable.
    global_step = tf.Variable(1, trainable=False, name='global_step')

    train_data = np.load(train_path)
    eval_data = np.load(eval_path)

    # Create the training model.
    with tf.name_scope("Train_model"):
        train_model = Model(5, 14, 1)  # TODO: use args_parser instead of hard-coding hyperparams of the model
        train_model.build_graph()

    # Create a copy of the training model for validation.
    with tf.name_scope("Eval_model"):
        eval_model = ... # TODO
        # eval_model.build_graph()

    # Count and print the number of trainable parameters.
    num_param = 0
    for v in tf.trainable_variables():
        print(v.name, str(v.get_shape()))
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))

    # Create the optimizer for the training model.
    train_model.optimization_routines()

    # Create the saver object to store checkpoints. We keep track of only 1 checkpoint.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)

    # Initialize the variables.
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())

    models = [train_model, eval_model]
    data = [train_data, eval_data]
    return models, data, saver, global_step, experiment_dir


def load_latest_checkpoint(sess, saver, experiment_dir):
    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError("could not load checkpoint")


def train():
    """
    The main training loop. Loads the data, creates the model, and trains for the specified number of epochs.
    """
    with tf.Session() as sess:

        # Create the models and load the data.
        models, data, saver, global_step, experiment_dir = create_model(sess, "train.npy", "eval.npy", "./experiments")
        train_model, valid_model = models
        train_data, valid_data = data
        print("Model created")

        # Training loop configuration.
        stop_signal = False
        epoch = 0
        train_loss = 0
        print("Running Training Loop.")

        # Initialize the data iterators.
        sess.run(train_model.iterator.initializer, feed_dict={train_model.train_placeholder: train_data})
        # sess.run(valid_iter.initializer) # TODO


        while not stop_signal:
            # Training.
            try:
                step_loss, step = train_model.step(sess)
                train_loss += step_loss

            except tf.errors.OutOfRangeError:
                print("epoch: ", epoch, " train loss: ", train_loss)
                train_loss = 0
                sess.run(train_model.iterator.initializer, feed_dict={train_model.train_placeholder: train_data})
                epoch += 1
                if epoch >= train_model.num_epochs:
                    stop_signal = True
                    break

            # TODO: evaluation pipeline

        print("Training Finished.")


if __name__ == "__main__":
    train()









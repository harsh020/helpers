import tempfile

import tensorflow as tf


class LRFinder:
    def __init__(self, model, lr_range=[1e-10, 1e1], beta=0.98, stop_factor=4):
        self.model = model
        self.lr_range = lr_range
        self.beta = beta
        self.stop_factor = stop_factor
        self.stop_training = False
        self.iterations = 0
        self.mvg_avg_loss = 0
        self.min_loss = 1e9
        self.lrs = []
        self.losses = []

    def _reset(self):
        self.stop_training = False
        self.iterations = 0
        self.mvg_avg_loss = 0
        self.min_loss = 1e9
        self.lrs = []
        self.losses = []

    def _scheduler(self, start_lr, end_lr, iterations):
        self.lr_factor = (end_lr / start_lr)**(1./iterations)

    def on_train_begin(self, logs=None):
        self._reset()

    def on_batch_end(self, batch, logs=None):
        self.iterations += 1

        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr*self.lr_factor)

        loss = logs['loss']
        self.mvg_avg_loss = (self.beta*self.mvg_avg_loss) + ((1-self.beta)*loss)
        smooth_loss = self.mvg_avg_loss / (1-(self.beta**self.iterations))
        self.losses.append(smooth_loss)

        stop_loss = self.stop_factor * self.min_loss
        if self.iterations > 1 and smooth_loss > stop_loss:
            self.stop_training = True

        if self.iterations == 0 or smooth_loss < self.min_loss:
            self.min_loss = smooth_loss
#         print(f'\nIterations: {self.iterations}, lr: {lr}, loss: {smooth_loss}/{loss}, lrf: {self.lr_factor}')

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_training:
            self.model.stop_training = True
            return

    def find(self, train_ds, epochs=None, steps_per_epoch=None, batch_size=32):
        if epochs is None:
            raise ValueError(f'Invalid value {epochs} for epochs')

        if steps_per_epoch is None:
            steps_per_epoch = len(train_ds)

        self._scheduler(self.lr_range[0], self.lr_range[1], steps_per_epoch*epochs)

        with tempfile.NamedTemporaryFile(prefix='init', suffix='.h5') as init_config:
            # save model config
            self.model.save_weights(init_config.name)
            init_lr = tf.keras.backend.get_value(self.model.optimizer.lr)

            lr_finder_cb = tf.keras.callbacks.LambdaCallback(
                on_train_begin= lambda logs: self.on_train_begin(logs),
                on_batch_end= lambda batch, logs: self.on_batch_end(batch, logs),
                on_epoch_end= lambda epoch, logs: self.on_epoch_end(epoch, logs)
            )

            self.model.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                           callbacks=[lr_finder_cb])

            # restore model config
            tf.keras.backend.set_value(self.model.optimizer.lr, init_lr)
            self.model.load_weights(init_config.name)

    def plot_loss(self, skip_begin=10, skip_end=1, title=""):
        lrs = self.lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")

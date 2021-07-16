import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        self.line1, = ax.plot(self.x, self.losses, 'r-')

        self.logs = []
        #plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        #f, (ax1, ax2) = plt.subplots(1, 1)

        self.line1.set_ydata(self.losses)
        self.line1.set_xdata(self.x)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

#        ax1.set_yscale('log')
#        ax1.plot(self.x, self.losses, label="loss")
#        ax1.plot(self.x, self.val_losses, label="val_loss")
#        ax1.legend()
#
 #       ax2.plot(self.x, self.acc, label="accuracy")
 #       ax2.plot(self.x, self.val_acc, label="validation accuracy")
 #       ax2.legend()

         #plt.show();

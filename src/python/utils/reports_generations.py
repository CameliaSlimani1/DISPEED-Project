""" Generation of reports """
import matplotlib.pyplot as plt


def plotPareto (x, y, x_label, y_label, color, rotation=0):
    fig = plt.figure(figsize=(12,10), dpi=80)
    plt.plot(x, y, color)
    plt.xticks(rotation=rotation)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plotPareto_2 (x1, x2, y1, y2, x_label, y_label, color1, color2, label1, label2, rotation=0):
    fig = plt.figure(figsize=(12,10), dpi=80)
    plt.plot(x1, y1, color1, label=label1)
    plt.plot(x2, y2, color2, label=label2)
    plt.xticks(rotation=rotation)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plotLosses (train_loss, val_loss, epochs, x_label, y_label, color1, color2,  rotation=0):
    fig = plt.figure(figsize=(12,10), dpi=80)
    plt.plot(epochs, train_loss, color1)
    plt.plot(epochs, val_loss, color2)
    plt.xticks(rotation=rotation)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
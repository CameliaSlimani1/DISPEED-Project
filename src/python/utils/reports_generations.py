""" Generation of reports """
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

def plotPareto (x, y, x_label, y_label, color, rotation=0):
    fig = plt.figure(figsize=(12,10), dpi=80)
    plt.plot(x, y, color)
    plt.xticks(rotation=rotation)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

#TODO : redesign this function
def plotPareto_2 (x1, x2, y1, y2, x_label, y_label, color1, color2, label1, label2, rotation=0):
    fig = plt.figure(figsize=(12,10), dpi=80)
    plt.plot(x1, y1, color1, label=label1)
    plt.plot(x2, y2, color2, label=label2)
    plt.xticks(rotation=rotation, fontsize=25)
    plt.yticks(rotation=rotation, fontsize=25)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
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

def plot3D(x, y, z1, z2):
    fig = plt.figure(figsize=(12,10), dpi=80)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    #fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    surf = ax.plot_trisurf(x, y, z1, linewidth=0, antialiased=False)
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    surf = ax.plot_trisurf(x, y, z2, linewidth=0, antialiased=False)
    ax.zaxis.set_major_formatter('{x:.02f}')
    plt.show()


def analyze_reports (filename):
    df = pd.read_csv(filename, sep=";")
    print(df.head())
    df.plot()



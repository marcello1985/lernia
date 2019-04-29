import matplotlib.pyplot as plt
import matplotlib

def style():
    plt.rcParams['axes.facecolor'] = 'white'
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'legend.fontsize': 12,'legend.handlelength': 2})
    plt.rcParams.update({'axes.labelsize': 12})
    plt.rcParams.update({'axes.titlesize': 14})

import matplotlib.pyplot as plt
import matplotlib

colorL = ["firebrick","sienna","olivedrab","crimson","steelblue","tomato","palegoldenrod","darkgreen","limegreen","navy","darkcyan","darkorange","brown","lightcoral","blue","red","green","yellow","purple","black"]

def style():
    plt.rcParams['axes.facecolor'] = 'white'
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    matplotlib.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'legend.fontsize': 12,'legend.handlelength': 2})
    plt.rcParams.update({'axes.labelsize': 12})
    plt.rcParams.update({'axes.titlesize': 14})

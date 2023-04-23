import matplotlib.pyplot as plt
import os
import json
import numpy as np



def plot_graph(x = None,y = [],title = "",xlabel = "",ylabel = ""):
    lens = [len(a) for a in y]
    if len(lens) == 1:
        if x == None:
            plt.plot(y)
        else :
            plt.plot(x,y)
    if len(lens) == 2:
        if x == None: # in that case x will be 0,1,2...
            plt.plot(y[0],'r--',y[1],'bs')
        else:
            plt.plot(x,y[0],'r--',x,y[1],'bs')
    if len(lens) == 3:
        if x == None: # in that case x will be 0,1,2...
            plt.plot(y[0],'r--',y[1],'bs',y[2],'g^')
        else:
            plt.plot('x',y[0],'r--','x',y[1],'bs','x',y[2],'g^')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()





def main():
    json_full_path = ""
    file = open(json_full_path)
    json_file = json.load(file)
    y = json_file[""]
    x = list(range(len(y)))
    items_i_want = ['fid_history','kl_losses',
                    'recon_losses','total_losses',
                    'end_epoch','fid_history']
    plot_graph(x,y,"title")


    file.close()


# import json
# def main():
#     json_full_path = ""
#     file = open(json_full_path)
#     json_file = json.load(file)
#     min_fid = min(json_file["fid_history"])

if __name__ == "__main__":
    main()
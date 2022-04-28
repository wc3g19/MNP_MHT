import plot as plt
import os
from matplotlib import pyplot
from os import listdir
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np


def get_files(root):
    '''
    Reccursivaly returns all files contained within the inputted root folder
    '''
    files = [join(root, f) for f in listdir(root) if isfile(
        join(root, f))]  # List of files in current search
    # List of direcectories within search
    dirs = [d for d in listdir(root) if isdir(join(root, d))]
    for d in dirs:
        # Reccursivaly calls get_files on all sub-folders
        files_in_d = get_files(join(root, d))
        for f in files_in_d:
            # Appends files list with new files in folder searched by get_files
            files.append(join(root, f))
    return files


def popcwd(filepath):
    '''
    Removes current working directory from string passed in and returns new name
    '''
    cwd = os.getcwd()
    filepath = filepath.replace(cwd + "\\" + file_name + '\\', '')
    return filepath


def get_particle_data(trial):
    '''
    Returns list of key data for a specific trial
    '''
    particle_data = []
    # Repeats for all files in global variable desired files
    for i in desired_files:
        i_list = []
        name = popcwd(i)
        area = float(plt.plotter(filepath=i).area(trial=trial))
        FWHM = np.array(plt.plotter(filepath=i).derivative_width(trial=trial))
        harmonic = np.array(plt.plotter(
            filepath=i).resolution(trial=trial, plot=False))
        i_list.append(name)  # Appends file name
        i_list.append(area)  # Appends area of given trial
        i_list.append(FWHM)  # Appends FWHM of given trial
        i_list.append(float(harmonic))  # Appends harmonic of trial
        # Appends all prev. data to running list for each file in desired data
        particle_data.append(i_list)

    particle_data = np.array(particle_data, dtype=object)
    return particle_data

def plot_data(trial, size_index, plotter="FWHM"):
    particle_data = get_particle_data(trial)
    xdat = particle_data[:, 1]
    if plotter == "FWHM":
        ydat = particle_data[:, 2]
        plot_label = "Full-Width Half-Maximum"
    if plotter == "harmonic":
        ydat = particle_data[:, 3]
        plot_label = 'M3 |X(freq)|'
    elif plotter == "AMF":
        AMFS = [180,150,120,90,60]
        ydat = [AMFS[trial[0]]]*len(xdat)
        plot_label = 'AMF Amplitude [mT]'
    #print(np.shape(xdat))
    #print(np.shape(ydat))

    #ymid = [abs((i[0] + i[1])/2) for i in ydat]
    #yerr = [abs(i[0]-i[1]) for i in ydat]

    plot_index = 0
    colour_index = 0
    marker_list = [".", "*", "x", "_", "^"]
    colour_list = ["r", "g", "b", "c", "m"]
    #size_list = [11, 9, 7, 5, 3]
    size_list = [70, 55, 40, 25, 10]
    for strength in range(5):
        marker_index = 0
        for cluster in range(4):
            #print(plot_index, marker_list[marker_index], 
            #      colour_list[colour_index], size_list[size_index])
            pyplot.scatter(xdat[plot_index], ydat[plot_index], marker=marker_list[marker_index], c = colour_list[colour_index], s=size_list[size_index])
            #pyplot.errorbar(xdat[plot_index], ymid[plot_index], yerr=yerr[plot_index]/2,
            #                fmt=marker_list[marker_index]+colour_list[colour_index], ms = size_list[size_index], elinewidth=1)
            marker_index += 1
            plot_index += 1
        colour_index += 1
    # [., *, x, _, ^, r, g, b, c, m, 3, 5, 7, 9, 11] = [0I, 2P, 3P, 6P, K1e5, K2e5, K3e4, K5e4, K7e4, Oe1800, Oe1500, Oe1200, Oe900, Oe600]
    pyplot.title("Derivative Width Vs. Area")
    pyplot.ylabel(plot_label)
    pyplot.xlabel("SAR [W/g]")


def SAR_AMF():
    global file_name
    file_name = "freq_3e+5_Ms450"
    all_files = get_files(os.getcwd() + "\\" + file_name)

    global desired_files
    desired_files = [i for i in all_files if i.endswith(
        "\\hystLoop_rea0.dat")]

    #files = get_files("freq_3e+5_Ms450")
    xdat = []
    ydat = []
    for i in ([0,0],[1,4],[2,4],[3,4],[4,4]):
        particle_data = get_particle_data(trial = i)
        part_dat = [j for j in particle_data if j[0] ==
                    'aniso_random_K3e4_Ms450\\clusters_2p\\hystLoop_rea0.dat'][0]
        x = part_dat[1]
        AMFS = [180, 150, 120, 90, 60]
        y = AMFS[i[0]]
        xdat.append(x)
        ydat.append(y)
        pyplot.scatter(x, y, marker = "*", color = "blue")
    
    xdat = []
    ydat = []
    for i in ([0, 0], [1, 4], [2, 4], [3, 4], [4, 4]):
        particle_data = get_particle_data(trial=i)
        part_dat = [j for j in particle_data if j[0] ==
                    'aniso_random_K2e5_Ms450\\clusters_2p\\hystLoop_rea0.dat'][0]
        x = part_dat[1]
        AMFS = [180, 150, 120, 90, 60]
        y = AMFS[i[0]]
        xdat.append(x)
        ydat.append(y)
        pyplot.scatter(x, y, marker="*", color="green")

        xdat = []
    ydat = []
    for i in ([0, 0], [1, 4], [2, 4], [3, 4], [4, 4]):
        particle_data = get_particle_data(trial=i)
        part_dat = [j for j in particle_data if j[0] ==
                    'aniso_random_K2e5_Ms450\\clusters_0I\\hystLoop_rea0.dat'][0]
        x = part_dat[1]
        AMFS = [180, 150, 120, 90, 60]
        y = AMFS[i[0]]
        xdat.append(x)
        ydat.append(y)
        pyplot.scatter(x, y, marker=".", color="green")
    
    xdat = []
    ydat = []
    for i in ([0, 0], [1, 4], [2, 4], [3, 4], [4, 4]):
        particle_data = get_particle_data(trial=i)
        part_dat = [j for j in particle_data if j[0] ==
                    'aniso_random_K1e5_Ms450\\clusters_2p\\hystLoop_rea0.dat'][0]
        x = part_dat[1]
        AMFS = [180, 150, 120, 90, 60]
        y = AMFS[i[0]]
        xdat.append(x)
        ydat.append(y)
        pyplot.scatter(x, y, marker="*", color="red")

    xdat = []
    ydat = []
    for i in ([0, 0], [1, 4], [2, 4], [3, 4], [4, 4]):
        particle_data = get_particle_data(trial=i)
        part_dat = [j for j in particle_data if j[0] ==
                    'aniso_random_K2e5_Ms450\\clusters_6p\\hystLoop_rea0.dat'][0]
        x = part_dat[1]
        AMFS = [180, 150, 120, 90, 60]
        y = AMFS[i[0]]
        xdat.append(x)
        ydat.append(y)
        pyplot.scatter(x, y, marker="_", color="green")
    
    pyplot.plot([100, 220], [60,60], color = "red", linestyle = "--", linewidth = 0.5, zorder=-5)
    pyplot.ylabel("AMF Amplitude [mT]")
    pyplot.ylim(0, 190)
    pyplot.xlabel("SAR [W/g]")

    
def bar_SAR_H(result = "SAR"):
    field_strengths = ['180 [mT]', '150 [mT]', '120 [mT]', '90 [mT]', '60 [mT]']
    d = {"Interaction":['0I', '2P', '3P', '6P']}
    if result == "SAR":
        ind = 1
        ylabel = "SAR [W/g]"
    elif result == "FWHM":
        ind = 2
        ylabel = "Full-Width Half-Maximum"
    elif result == "harmonic":
        ind = 3
        ylabel = 'M3 |X(freq)|'

    for i in range(5):
        if i == 0:
            particle_data = get_particle_data([0, 0])
        else:
            particle_data = get_particle_data([i, 4])

        if ind==1 or ind==3:
            ydat = particle_data[:,ind]
        else:
            ydat = [sum(i)/2 for i in particle_data[:, ind]]
        k = 1  
        d[field_strengths[i]] = [ydat[1*4 + i] for i in range(4)]
    #   k = 0 (1e5), 1 (2e5), 2 (3e4), 3 (5e4), 4 (7e4)
    df = pd.DataFrame(d)
    print(df)

    df.set_index("Interaction").plot(kind='bar', align='center',
                                     width=0.6).legend(bbox_to_anchor=(1.0, 1.0))
    pyplot.xticks(rotation=0)
    pyplot.ylabel(ylabel)


def bar_SAR_aniso(trial, result="SAR"):
    particle_data = get_particle_data(trial)


    if result == "SAR":
        ind = 1
        ylabel = "SAR [W/g]"
    elif result == "FWHM":
        ind = 2
        ylabel="Full-Width Half-Maximum"
    elif result == "harmonic":
        ind = 3
        ylabel = 'M3 |X(freq)|'

    if ind == 1 or ind == 3:
            ydat = particle_data[:, ind]
    else:
        ydat = [sum(i)/2 for i in particle_data[:, ind]]

    labels = ['1e5 [erg/cc]', '2e5 [erg/cc]',
              '3e4 [erg/cc]', '5e4 [erg/cc]', '7e4 [erg/cc]']
    label2 = ['0I', '2P', '3P', '6P']

    red = [ydat[0*4 + i] for i in range(4)]
    green = [ydat[1*4 + i] for i in range(4)]
    magenta = [ydat[2*4 + i] for i in range(4)]
    cyan = [ydat[3*4 + i] for i in range(4)]
    blue = [ydat[4*4 + i] for i in range(4)]

    d = {"Interaction": label2, labels[0]: red, labels[1]: green, labels[2]: magenta, labels[3]:cyan, labels[4]:blue}
    df = pd.DataFrame(d)

    df.set_index("Interaction").plot(kind='bar', align='center',
                                     width=0.6).legend(bbox_to_anchor=(1.0, 1.0))
    pyplot.xticks(rotation = 0)
    pyplot.ylabel(ylabel)


global file_name    # Makes file_name a global var so can be accessed throughout script.py
file_name = "freq_3e+5_Ms450"   # Defines name of folder the data is located in
all_files = get_files(os.getcwd() + "\\" + file_name)        # Runs get_files function on file freq_...
desired_files = [i for i in all_files if i.endswith("\\hystLoop_rea0.dat")]   # Only selects files from folder 0I which end in .dat


#instance = plt.plotter(filepath="C:\\Users\\Joe's Laptop\\Desktop\\Uni\\Year 3\\IP\\Code\\freq_3e+5_Ms450\\aniso_random_K2e5_Ms450\\clusters_6P\hystLoop_rea0.dat")
#print(instance.file_info())
#instance.derivative_width(trial=[0,0], plot=True)
#print(instance.area([0,0]))
#print(instance.area([4,4]))

#instance.plot([0, 0], color="red", linestyle="--", linewidth=0.75)
#instance.plot([4,4])
#pyplot.show()
#instance.resolution()
#pyplot.show()
#instance.file_info()
#instance.plot([0, 0], show=False)
#instance.plot([4, 4], show=True)
#print(instance.area([0,0]))
#print(instance.area([4,4]))
#print(instance.derivative_width(trial=[2,4]))
#plot_data([4,4], 0)
#pyplot.show()

# *** SAR colour field strength ***
'''
bar_SAR_H(result="SAR")
pyplot.savefig(
    "C:\\Users\\Joe's Laptop\\Desktop\\Uni\\Year 3\\IP\\Plots\\derivative width vs area\\SARBarH.png", dpi=500, bbox_inches='tight')
'''
# *** SAR colour aniso ***
'''
bar_SAR_aniso([4,4], result="SAR")
pyplot.savefig(
    "C:\\Users\\Joe's Laptop\\Desktop\\Uni\\Year 3\\IP\\Plots\\derivative width vs area\\SARBar.png", dpi=500, bbox_inches='tight')
'''

# *** SAR-MPI ***
'''
plotter = "AMF"
for i in range(5):
    if i == 0:
        plot_data([i, 0], i, plotter)
        pyplot.savefig(
            "C:\\Users\\Joe's Laptop\\Desktop\\Uni\\Year 3\\IP\\Plots\\derivative width vs area\\"+str(i)+"0.png", dpi=500)
    else:
        plot_data([i, 4], i, plotter)
        pyplot.savefig("C:\\Users\\Joe's Laptop\\Desktop\\Uni\\Year 3\\IP\\Plots\\derivative width vs area\\"+str(i)+"4.png", dpi=500)
    #pyplot.close()     # Use if trying to create individual plots
'''

# *** SAR-AMF ***
'''
SAR_AMF()
pyplot.show()
'''
#pyplot.show()

#headders = ["File Path", "Area", "derivative widths [avg., gaus.]"]     # Assigns headder names to data
#df = pd.DataFrame(particle_data, columns=headders)      # Creates pandas df of particle data array
#df.to_csv('out.csv', header=False, index=False)         # Saves pandas df as out.csv file

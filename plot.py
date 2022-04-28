from extract_data import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft


class plotter(data):
    def __init__(self, filename=None, filepath=None):
        # Inherets all attributes and methods from extrax_data.py
        if filename != None:
            super().__init__(filename=filename)
        elif filepath != None:
            super().__init__(filepath=filepath)
        else:
            raise ValueError("no data file path given")

    def plot(self, trial=[0, 0], show=False, **kwargs):
        '''
        plots x, y data for given trial [mag. val., cycle no.] can give 2D list to plot multiple trials on same axis. **kwargs used to pass
        additional formatting into matplotlib.plot if requred
        '''
        # single() and is2Dlist() handels case where attempting to plot two trials
        def single(i):
            '''
            Only used within plot()
            '''
            self.get_data(trial)
            plt.plot(self.x*10**3, self.y, **kwargs)

        def is2DList(matrix_list):
            '''
            Only used within plot()
            '''
            if isinstance(matrix_list[0], list):
                return True
            else:
                return False

        if is2DList(trial):
            [single(i) for i in trial]
        else:
            single(trial)

        plt.xlabel("Field H[mT]")
        plt.ylabel("Magnetisation [M]")

        if show:
            plt.show()

    def get_data(self, trial=[0, 0]):
        self.run_data = np.array(self.desired_data(trial))
        self.x = np.delete(self.run_data[:, 4], 0)*(1/10000)
        self.y = np.delete(self.run_data[:, 5], 0)

    def area(self, trial=[0, 0]):
        '''
        Returns SAR of hysteresis loop given constants defined below, Applies Eq.(2) from report to calc. SAR
        '''
        f = 3*10**5
        Ms = 450 * (10**(-4))
        rho = 5.2       # g/m^3 Ruta et al.
        self.get_data(trial)  # Retrieves data for given trial
        A = abs(simps(self.x, self.y))
        SAR = A*f*Ms/rho
        return SAR

    def moving_average(self, dydx, avg_size):
        moving_dydx = []
        i = 0
        while i < len(dydx) - avg_size + 1:
            # Sums all dydx values in window
            window_sum = sum(dydx[i: i + avg_size])
            moving_average = window_sum/avg_size            # Calculates average of window
            # Adds average to moving_dydx list
            moving_dydx.append(moving_average)
            i += 1
        return moving_dydx

    def gauss(self, x, H, A, x0, sigma):
        '''
        Defines gaussian function, given x and constants will return y
        '''
        return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def gauss_fit(self, x, y):
        '''
        Calculates closest fitting gaussian function to x, y data
        '''
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        # scipy.curve_fit optimises arguments to best fit func. to data, initial estimates are given from vars. defined above
        popt, _ = curve_fit(self.gauss, x, y, p0=[min(y), max(y), mean, sigma])
        return popt

    def find_nearest(self, array, value):
        '''
        given array of data finds closest index to a value
        '''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def half_width(self, x, y, plot=False):
        '''
        Calculates FHWM of derivative data
        '''
        x = np.array(x)
        y = np.array(y)
        min_y = np.min(y)
        max_y = np.max(y)

        half_min_max = (min_y+max_y)/2      # Finds half position of x, y data
        # Finds index of max value in curve
        split = np.where(y == max_y)[0][0]
        first_halfy = y[:split]
        second_halfy = y[split:]            # Splits cuve about max value
        first_halfx = x[:split]
        second_halfx = x[split:]

        # Finds x value at the half point of first curve
        xone = first_halfx[self.find_nearest(first_halfy, half_min_max)]
        # Finds x value at the half point of second curve
        xtwo = second_halfx[self.find_nearest(second_halfy, half_min_max)]
        if plot:
            plt.plot(x*1000, y, color='red', zorder=10)
            #plt.plot(second_halfx*1000, second_halfy, color='red', zorder=10)
            #plt.plot(first_halfx*1000, first_halfy, color='red', zorder=10)
            plt.scatter(xone*1000, half_min_max, color='cyan', zorder=100)
            plt.scatter(xtwo*1000, half_min_max, color='cyan', zorder=100)
            plt.xlabel("Field H[mT]")
            plt.ylabel("Gradient [AT/m]")

        # Returns width at half point
        return abs(xone-xtwo)

    def derivative_width(self, trial=[0, 0], avg=True, gaus=True, plot=False):
        widths = [0, 0]
        self.get_data(trial)

        # Splits self.y,self.x into one way hysteresis curve
        halfy = self.y[:int(len(self.y)/2)]
        halfx = self.x[:int(len(self.y)/2)]

        # Calculates y differences (magnetisation)
        dy = [i-j for j, i in zip(halfy[:-1], halfy[1:])]
        # Calculates x differences (field strength)
        dx = [i-j for j, i in zip(halfx[:-1], halfx[1:])]
        # Calculates gradient of magnetisation vs. field strength
        dydx = [i/j for i, j in zip(dy, dx)]

        # Datapoints are lost in process of finding derivative so x points must be estimated at derivative positions
        # Calculates x at half way points in same position as gradients are taken
        interpolatex = [(i+j)/2 for j, i in zip(halfx[:-1], halfx[1:])]

        avg_size = 7        # Size of moving average window
        moving_dydx = self.moving_average(dydx, avg_size)
        slicer = int(avg_size/2)

        interpolatex = np.array(interpolatex)
        dydx = np.array(dydx)

        if plot:
            plt.plot(interpolatex*1000, dydx, zorder=0)

        # Finds y poinnts of closest gaussian curve
        y_eval = self.gauss(
            interpolatex, *self.gauss_fit(interpolatex, dydx))

        if avg:
            # Adds FHWM from moving average
            half_width_avg = self.half_width(
                interpolatex[slicer:-slicer], moving_dydx, plot=False)
            widths[0] = half_width_avg

        if gaus:
            # Adds FWHM from closest gaussian
            half_width_gaus = self.half_width(
                interpolatex, y_eval, plot=False)
            widths[1] = half_width_gaus

        return widths       # Returns list of two FWHM results

    def area_vs_trial(self):
        for index, i in enumerate(self.complete_list):
            y = []
            first_area = self.area([index, 0])
            for j in i:
                trial = [index, j-1]
                y.append(self.area(trial))
            y = np.array(y)
            x = np.arange(1, len(y) + 1)
            plt.scatter(x, y)
        plt.show()

    def resolution(self, trial=[0, 0], freq=3.0e5, plot=False):
        '''
        Performs Fourier transform to data from given trial. plots if required.
        '''
        self.get_data(trial)
        samplesize = np.linspace(0, 1/freq, len(self.y))
        firsthalfy = self.y[:int(len(self.y)/2)]
        secondhalfy = self.y[int(len(self.y)/2):]
        total = np.append(firsthalfy, secondhalfy)

        sr = len(total)*freq
        fty = fft(total)
        fty_plot = np.abs(fty)

        N = len(fty)
        n = np.arange(N)
        T = N/sr
        freq = n/T

        if plot:
            plt.subplot(131)
            plt.plot(samplesize, total)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')

            plt.subplot(132)
            plt.stem(freq, fty_plot, 'b',
                     markerfmt=" ", basefmt="-b")
            plt.xlabel('Freq (Hz)')
            plt.ylabel('FFT Amplitude |X(freq)|')
            plt.xlim(0, 10000000)

            plt.subplot(133)
            plt.plot(samplesize, ifft(fty), 'r')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.show()

        return(fty_plot[3])     # Returns third harmonic

import os
import pandas as pd

class data():
    def __init__(self, file_name = None, filepath=None):
        '''
        Extracts positional data from instance
        '''
        if file_name != None:       # If file name in current directory given
            dat_path_defult = os.path.dirname(os.path.abspath(__file__))                # Locates directory program is in and follows to "Data" folder
            self.data_path = dat_path_defult + file_name 
        elif filepath != None:      # Else if file path is given
            self.data_path = filepath
            
        with open(self.data_path, 'r') as data:
            mag_trial_index = {}    # dictionary of list
            cycle_dict = {}
            complete_list = []  # compiled list of trails
            index_start = 0
            
            cycle = 0
            mag_trial = 0
            
            cycle_offset = 0

            count = sum(1 for line in open(self.data_path))
            
            for index, line in enumerate(data):
                lineList = line.split("\t")                             # Splits all datapoints on tab into list
                if int(lineList[1]) != cycle or index == count-1:                           # If the cycle number has changed take note of start and end points
                    dictKey = int(lineList[1]) - int(cycle_offset)
                    cycle_dict[dictKey] = [index_start, index-1]          # Creates dictionary {cycle number, [start line, end line]}
                    mag_trial_index.update(cycle_dict)                        # Adds to dictionary
                    index_start = index
                    cycle += 1
                    if int(lineList[0]) != mag_trial or index == count:               # If on different magnetisation setting append dictionary to list
                        complete_list.append(cycle_dict)
                        cycle_offset = lineList[1]
                        cycle_dict = {}
                        mag_trial += 1
            self.complete_list = complete_list

    def desired_data(self, run):
        '''
        Returns raw data from given instance and the run you would like [magnetisation, cycle no.]
        '''
        run_data = []
        lines = self.complete_list[run[0]].get(run[1]+1)
        with open(self.data_path, 'r') as data:
            for index, line in enumerate(data):            
                #print(lines)
                if index in range(lines[0], lines[1]+1):      # If the index is within the bouds of desired cycle
                    line = line.split("\t")                 # Creates list from line in txt
                    for i, j in enumerate(line):            # Formats list so all numbers are floats and removes new line character
                        if i == len(j)-1:
                            j = j.replace("\n", "")
                        line[i] = float(j)
                    run_data.append(line)
        return run_data

    def file_info(self, returnPd=False):
        '''
        prints positional information about instance data, if reutnrPd=True will return the panda dataframe
        '''
        df = pd.DataFrame(self.complete_list)
        df = df.rename_axis(index='Mag. Value', columns='CYCLE')    # Renames columns and rows
        ind = []
        for i in range(len(self.complete_list)):
            ind.append(int(self.desired_data([i, 0])[0][4]))      # Appends ind with max H value of test
        df = df.set_axis(ind, axis='index')                         # Sets row indecies as max H valuse
        if returnPd == True:
            return df
        else:
            print(df)

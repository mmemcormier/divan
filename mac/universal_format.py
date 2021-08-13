import numpy as np
import pandas as pd
from os import path
from scipy.integrate import simps
import re
from argparse import ArgumentParser
from datetime import datetime
import regex as re
from reader_new import ParseNeware
import streamlit as st

CYC_TYPES = {'charge', 'discharge', 'cycle'}
RATES = np.array([1/160, 1/80, 1/40, 1/20, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 10])
C_RATES = ['C/160', 'C/80', 'C/40', 'C/20', 'C/10', 'C/5', 'C/4', 'C/3', 'C/2', '1C', '2C', '3C', '4C', '5C', '10C']
FILE_TYPES = ["Neware", "Novonix"]

class UniversalFormat():
    def __init__(self, genericfile, all_lines=None, ref_cap=None):
        ## Parse file to determine what kind of file it is

        if all_lines is not None:
            lines = []
            self.genericfile = genericfile
            for line in all_lines:
                lines.append(line.decode('unicode_escape'))
        else:
            self.genericfile = genericfile[:-4]
            with open(genericfile, 'r') as f:
                lines = f.readlines()

        if "Cycle ID" == lines[0][:8]:
            self.file_type = FILE_TYPES[0]
            self.neware = ParseNeware(self.genericfile, all_lines=lines)
            self.formatted_file = self.neware.get_universal_format()
            rec = self.neware.get_rec()

        else:

            self.file_type = FILE_TYPES[1]
            header_list = lines[12].split(",")
            header_list[-1] = header_list[-1].strip()
            mass = float(str(lines[4]).split(" ")[2])
            self.formatted_file = pd.read_csv(genericfile, skiprows=range(0,14), header=0, names=header_list)
            
        C_rates = []
        tot_cell_cap_ind = self.formatted_file["Capacity (Ah)"].idxmax()
        tot_cell_cap = self.formatted_file["Capacity (Ah)"].values.tolist()[tot_cell_cap_ind]
        for i in self.formatted_file["Meas I (A)"].values.tolist():
            if i == 0:
                C_rates.append(None)
            else:
                rate = tot_cell_cap/abs(i)

                if rate is not None and rate != 0:
                    for ind in range(len(RATES)):
                        if ind == 0:
                            if 1/rate < float(RATES[0]):
                                c_rate = C_RATES[0]
                                break
                        elif ind == len(RATES) - 1:
                            if 1/rate > float(RATES[-1]):
                                c_rate = C_RATES[-1]
                                break
                        elif (1/rate > float((RATES[ind - 1] + RATES[ind]) / 2)) and (1/rate < float((RATES[ind + 1] + RATES[ind]) / 2)):
                            c_rate = C_RATES[ind]                    
                            break

                else:
                    c_rate = None

                C_rates.append(c_rate)


        self.formatted_file["C_rate"] = C_rates


        self.chg_crates = []
        self.dis_crates = []
        
        steps = self.formatted_file["Prot.Step"].unique().tolist()
        
        grouped = self.formatted_file.groupby(["Prot.Step"])


        for step in steps:
            group = grouped.get_group(step)
            if step == 1 or step == 5:
                if group["C_rate"].values.tolist()[-1] not in self.chg_crates and group["C_rate"].values.tolist()[-1] is not None:
                    self.chg_crates.append(group["C_rate"].values.tolist()[-1])
            elif step == 2 or step == 6:
                if group["C_rate"].values.tolist()[-1] not in self.dis_crates and group["C_rate"].values.tolist()[-1] is not None:
                    self.dis_crates.append(group["C_rate"].values.tolist()[-1])
            
    def get_ncyc(self):
        '''
        Returns the total number of cycles.
        '''
        return int(self.formatted_file['Cycle'].values[-1])

    def get_rates(self, cyctype='cycle'):
        if cyctype not in CYC_TYPES:
            raise ValueError('cyctype must be one of {0}'.format(CYC_TYPES))
        if cyctype == 'charge':
            return self.chg_crates
        elif cyctype == 'discharge':
            return self.dis_crates
        elif cyctype == 'cycle':
            rates = self.chg_crates
            for r in self.dis_crates:
                if r not in rates:
                    rates.append(r)
            return rates
        
    def select_by_rate(self, rate, cyctype='cycle'):
        '''
        Return record data for all cycles that have a particular rate.
        rate: dtype=string.
        cyctype: {'cycle', 'charge', 'discharge'}
        '''
        
        if rate not in C_RATES:
            raise ValueError('rate must be one of {0}'.format(C_RATES))

        if cyctype not in CYC_TYPES:
            raise ValueError('cyctype must be one of {0}'.format(CYC_TYPES))

        selected_cycs = []
        cycles = self.formatted_file.loc[self.formatted_file['C_rate'] == rate]
        cycnums = cycles['Cycle'].unique()
        for i in range(len(cycnums)):
            stepnums = cycles.loc[cycles['Cycle'] == cycnums[i]].values
            if cyctype == 'cycle':
                if len(stepnums) >= 2:
                    selected_cycs.append(cycnums[i])

            elif cyctype == 'charge':
                step = self.formatted_file.loc[self.formatted_file['Prot.Step'] == stepnums[0]]
                if step['C_rate'].values == rate:
                    selected_cycs.append(cycnums[i])

            elif cyctype == 'discharge':
                step = self.formatted_file.loc[self.formatted_file['Prot.Step'] == stepnums[-1]]
                if step['C_rate'].values == rate:
                    selected_cycs.append(cycnums[i])

# For cyctype='cycle' need to check that first and last step both have same rate.
# For charge/discharge need to check that first/last step are at C_rate=rate

        return selected_cycs

    
    def get_vcurve(self, cycnum=-1, cyctype='cycle'):

        #TODO
        # It looks like the first cycle was the issue, this is a temporary fix
        if cycnum == 1:
            cycnum = -1

        if cyctype not in CYC_TYPES:
            raise ValueError('cyctype must be one of {0}'.format(CYC_TYPES))

        if cycnum == -1:
            cycnum = self.get_ncyc() - 1

        try:
            cycle = self.formatted_file.loc[self.formatted_file['Cycle'] == cycnum]
        except:

            print('Cycle {} does not exist. Input a different cycle number.'.format(cycnum))

        if cyctype == 'charge':
            chg = cycle.loc[(cycle['Step'] == 1) | (cycle['Step'] == 5)]

            if len(chg) != 0:
                voltage = chg['Potential (V)'].values
                capacity = chg['Capacity_Density'].values / 1000
            else:
                return None, None

        elif cyctype == 'discharge':
            dis = cycle.loc[cycle['Step'] == 2]

            if len(dis) != 0:
                voltage = dis['Potential (V)'].values
                capacity = dis['Capacity_Density'].values / 1000
            else:
                return None, None

        elif cyctype == 'cycle':

            chg = cycle.loc[(cycle['Step'] == 1) | (cycle['Step'] == 5)]

            if len(chg) != 0:

                Vchg = chg['Potential (V)'].values
                Cchg = chg['Capacity_Density'].values

                dis = cycle.loc[cycle['Step'] == 2]
                Vdchg = dis['Potential (V)'].values
                Cdchg = dis['Capacity_Density'].values

                voltage = np.concatenate((Vchg, Vdchg)) / 1000
                capacity = np.concatenate((Cchg, -Cdchg+Cchg[-1])) / 1000

            else:
                st.write(cycnum)
                return None, None

        return capacity, voltage
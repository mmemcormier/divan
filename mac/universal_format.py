import numpy as np
import pandas as pd
from os import path
from scipy.integrate import simps
import re
from argparse import ArgumentParser
from datetime import datetime
from neware_parser import ParseNeware
import streamlit as st

CYC_TYPES = {'charge', 'discharge', 'cycle'}
RATES = np.array([1/160, 1/80, 1/40, 1/20, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 10])
C_RATES = ['C/160', 'C/80', 'C/40', 'C/20', 'C/10', 'C/5', 'C/4', 'C/3', 'C/2', '1C', '2C', '3C', '4C', '5C', '10C']
FILE_TYPES = ["Neware", "Novonix"]

class UniversalFormat():

    def __init__(self, genericfile, all_lines=None):
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
            self.formatted_df = self.neware.get_universal_format()

        else:

            self.file_type = FILE_TYPES[1]
            header_list = lines[12].split(",")
            header_list[-1] = header_list[-1].strip()
            mass = float(str(lines[4]).split(" ")[2])
            self.formatted_df = pd.read_csv(genericfile, skiprows=range(0,14), header=0, names=header_list)

        cap = self.formatted_df["Capacity (Ah)"].values
        max_inds = np.argpartition(cap, -5)[-5:]
        ref_cap = np.sum(cap[max_inds]) / 5
        cycnums = np.arange(1, 1 + self.get_ncyc())
        
        stepnums = self.formatted_df["Prot.Step"].unique()
        cycnums = np.zeros(len(stepnums), dtype='int')
        nstep = len(stepnums)
        step_rates = ['N/A']*nstep
        chg_rates = []
        chg_crates = []
        dis_rates = []
        dis_crates = []
        for i in range(len(stepnums)):
            step = self.formatted_df.loc[self.formatted_df["Prot.Step"] == stepnums[i]]
            cycnum = step["Cycle"].unique()
            cycnums[i] = cycnum[0]
            
            if step["Step"].unique()[0] in [1, 5]:
                chg_cur = step["Meas I (A)"].values
                chg_cur_max = np.amax(np.absolute(chg_cur))
                if chg_cur_max > 0.0:
                    cr = chg_cur_max / ref_cap
                    ind = np.argmin(np.absolute(RATES - cr))
                    chgrate = RATES[ind]
                    if chgrate not in chg_rates:
                        chg_rates.append(chgrate)
                        chg_crates.append(C_RATES[ind])
                    step_rates[i] = C_RATES[ind]
                
            if step["Step"].unique()[0] in [2, 6]:
                dis_cur = step["Meas I (A)"].values
                dis_cur_max = np.amax(np.absolute(dis_cur))
                if dis_cur_max > 0.0:
                    cr = dis_cur_max / ref_cap
                    ind = np.argmin(np.absolute(RATES - cr))
                    disrate = RATES[ind]
                    if disrate not in dis_rates:
                        dis_rates.append(disrate)
                        dis_crates.append(C_RATES[ind])
                    step_rates[i] = C_RATES[ind]
                
        self.step_df = pd.DataFrame()
        self.step_df["Cycle"] = cycnums
        self.step_df["Prot.Step"] = stepnums
        self.step_df["C_rates"] = step_rates
        print('Found charge C-rates: {}'.format(chg_crates))
        print('Found discharge C-rates: {}'.format(dis_crates))
        self.chg_crates = chg_crates
        self.dis_crates = dis_crates

    def get_ncyc(self):
        '''
        Returns the total number of cycles.
        '''
        return int(self.formatted_df['Cycle'].values[-1])

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
        cycles = self.step_df.loc[self.step_df["C_rates"] == rate]
        cycnums = cycles['Cycle'].unique()
        for i in range(len(cycnums)):
            stepnums = cycles.loc[cycles['Cycle'] == cycnums[i]].values
            if cyctype == 'cycle':
                if len(stepnums) >= 2:
                    selected_cycs.append(cycnums[i])

            elif cyctype == 'charge':
                step = self.step_df.loc[(self.step_df["Prot.Step"] == 1) | (self.step_df["Prot.Step"] == 5)]
                #step = self.formatted_df.loc[self.formatted_df['Prot.Step'] == stepnums[0]]
                if step['C_rate'].values == rate:
                    selected_cycs.append(cycnums[i])

            elif cyctype == 'discharge':
                step = self.step_df.loc[(self.step_df["Prot.Step"] == 2) | (self.step_df["Prot.Step"] == 6)]
                #step = self.formatted_df.loc[self.formatted_df['Prot.Step'] == stepnums[-1]]
                if step['C_rate'].values == rate:
                    selected_cycs.append(cycnums[i])

        return selected_cycs

    
    def get_vcurve(self, cycnum=-1, cyctype='cycle', active_mass=None):

        #TODO
        # It looks like the first cycle was the issue, this is a temporary fix
        if cycnum == 1:
            cycnum = -1

        if cyctype not in CYC_TYPES:
            raise ValueError('cyctype must be one of {0}'.format(CYC_TYPES))

        if cycnum == -1:
            cycnum = self.get_ncyc() - 1

        try:
            cycle = self.formatted_df.loc[self.formatted_df['Cycle'] == cycnum]
        except:

            print('Cycle {} does not exist. Input a different cycle number.'.format(cycnum))

        if cyctype == 'charge':
            chg = cycle.loc[(cycle['Step'] == 1) | (cycle['Step'] == 5)]

            if len(chg) != 0:
                # Wasn't working when I performed this check in neware_parser.py, this is a temporary fix until I
                #   can figure it out. better ask Marc!
                if max(chg['Potential (V)'].values) > 1000:
                    voltage = chg['Potential (V)'].values / 1000
                else:
                    voltage = chg['Potential (V)'].values / 1000

                capacity = chg['Capacity (Ah)'].values
            else:
                return None, None

        elif cyctype == 'discharge':
            dis = cycle.loc[cycle['Step'] == 2]

            if len(dis) != 0:
                if max(dis['Potential (V)'].values) > 1000:
                    voltage = dis['Potential (V)'].values / 1000
                else:
                    voltage = dis['Potential (V)'].values

                capacity = dis['Capacity (Ah)'].values
            else:
                return None, None

        elif cyctype == 'cycle':

            chg = cycle.loc[(cycle['Step'] == 1) | (cycle['Step'] == 5)]

            if len(chg) != 0:

                if max(chg['Potential (V)'].values) > 1000:
                    Vchg = chg['Potential (V)'].values / 1000
                else:
                    Vchg = chg['Potential (V)'].values

                Cchg = chg['Capacity (Ah)'].values


                dis = cycle.loc[cycle['Step'] == 2]
                if max(chg['Potential (V)'].values) > 1000:
                    Vdchg = chg['Potential (V)'].values / 1000
                else:
                    Vdchg = chg['Potential (V)'].values
                Cdchg = dis['Capacity (Ah)'].values

                voltage = np.concatenate((Vchg, Vdchg))
                capacity = np.concatenate((Cchg, -Cdchg+Cchg[-1]))

            else:
                st.write(cycnum)
                return None, None

        return capacity, voltage

    def get_dQdV(self, cycnum=-1, cyctype='cycle'):
        '''
        Get dQdV for specific cycle. Returns charge and discharge together.
        '''
        cchg, vchg = self.get_vcurve(cycnum=cycnum, cyctype='charge')
        cend = cchg[-1]
        delta_cchg = cchg[1:] - cchg[:-1]
        delta_vchg = vchg[1:] - vchg[:-1]
        inf_inds = np.where(np.absolute(delta_vchg) < 1e-12)
        vchg = np.delete(vchg, inf_inds[0] + 1)
        cchg = np.delete(cchg, inf_inds[0] + 1)
        dQdVchg = (cchg[1:] - cchg[:-1]) / (vchg[1:] - vchg[:-1])

        cdchg, vdchg = self.get_vcurve(cycnum=cycnum, cyctype='discharge')
        cdchg = -cdchg + cend
        delta_cdchg = cdchg[1:] - cdchg[:-1]
        delta_vdchg = vdchg[1:] - vdchg[:-1]
        inf_inds = np.where(np.absolute(delta_vdchg) < 1e-12)
        vdchg = np.delete(vdchg, inf_inds[0] + 1)
        cdchg = np.delete(cdchg, inf_inds[0] + 1)
        dQdVdchg = -(cdchg[1:] - cdchg[:-1]) / ((vdchg[1:] - vdchg[:-1]))

        voltage = np.concatenate((vchg[1:], vdchg[:-1]))
        dQdV = np.concatenate((dQdVchg, dQdVdchg))


        return voltage, dQdV
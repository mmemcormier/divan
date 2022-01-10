import numpy as np
import pandas as pd
from os import path
from scipy.integrate import simps
import re
from argparse import ArgumentParser
from datetime import datetime
from neware_parser import ParseNeware
#import streamlit as st

CYC_TYPES = {'charge', 'chg', 'discharge', 'dis', 'cycle', 'cyc'}
RATES = np.array([1/160, 1/80, 1/40, 1/20, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 10])
C_RATES = ['C/160', 'C/80', 'C/40', 'C/20', 'C/10', 'C/5', 'C/4', 'C/3', 'C/2', '1C', '2C', '3C', '4C', '5C', '10C']
FILE_TYPES = ["Neware", "Novonix"]

class UniversalFormat():

    def __init__(self, genericfile, all_lines=None, ref_cap=None):
        ## Parse file to determine what kind of file it is
        
        self.ref_cap = ref_cap
        
        if all_lines is not None:
            lines = []
            self.genericfile = genericfile
            for line in all_lines:
                if genericfile == "txt":
                    lines.append(line.decode('unicode_escape'))
                else:
                    lines.append(line)
        else:
            self.genericfile = genericfile[:-4]
            with open(genericfile, 'r', encoding='unicode_escape') as f:
                lines = f.readlines()

        if "Cycle ID" == lines[0][:8]:
            self.file_type = FILE_TYPES[0]
            parsed_data = ParseNeware(self.genericfile, all_lines=lines)
            self.formatted_df = parsed_data.get_universal_format()
            self.cap_type = parsed_data.cap_type
            
        else:

            self.file_type = FILE_TYPES[1]
            self.cap_type = "cum"
            
            nlines = len(lines)
            headlines = []
            for i in range(nlines):
                headlines.append(lines[i])
                l = lines[i].strip().split()
                if l[0][:6] == '[Data]':
                    hline = lines[i+1]
                    nskip = i+1
                    break
            
            header = ''.join(headlines)
            
            # find mass and theoretical cap using re on header str
            m = re.search('Mass\s+\(.*\):\s+(\d+)?\.\d+', header)
            m = m.group(0).split()
            mass_units = m[1][1:-2]
            if mass_units == 'mg':
                self.mass = float(m[-1]) / 1000
            else:
                self.mass = float(m[-1])
            
            if self.ref_cap is None:
                m = re.search('Capacity\s+(.*):\s+(\d+)?\.\d+', header)
                m = m.group(0).split()
                cap_units = m[1][1:-2]
                if cap_units == 'mAHr':
                    self.ref_cap = float(m[-1]) / 1000
                else:
                    self.ref_cap = float(m[-1])
                
            m = re.search('Cell: .+?(?=,|\\n)', header)
            m = m.group(0).split()
            self.cellname = m[-1]
            
            cols = hline.strip().split(",")
            self.formatted_df = pd.DataFrame([r.split(",") for r in lines[nskip+1:]],
                                             columns=cols)
            self.formatted_df.pop("Date and Time")
            
            
            self.formatted_df.rename(columns={'Capacity (Ah)': 'Capacity',
                                              'Potential (V)': 'Potential',
                                              'Run Time (h)': 'Time',
                                              'Time (h)': 'Time',
                                              'Current (A)': 'Current',
                                              'Cycle Number': 'Cycle',
                                              'Meas I (A)': 'Current',
                                              'Step Type': 'Step',
                                              'Step Number': 'Step'},
                                    inplace=True)
            print(self.formatted_df.columns)
            # Add Prot_step column even if step num exists.
            s = self.formatted_df.Step
            self.formatted_df["Prot_step"] = s.ne(s.shift()).cumsum() - 1
            self.formatted_df = self.formatted_df.apply(pd.to_numeric) 
            
            
        t = self.formatted_df["Time"].values
        dt = t[1:] - t[:-1]
        inds = np.where(dt <= 0.0)[0]
        #if len(inds) > 0:
            #print("Removing indices due to time non-monotonicity: {}".format(inds))
        self.formatted_df = self.formatted_df.drop(inds+1)
        inds = self.formatted_df.index[self.formatted_df["Potential"] < 0.0].tolist()
        self.formatted_df = self.formatted_df.drop(inds)
        
        
        cap = self.formatted_df["Capacity"].values
        max_inds = np.argpartition(cap, -5)[-5:]
        if self.ref_cap is None:
            self.ref_cap = np.sum(cap[max_inds]) / 5
            print("WARNING: Using {0:.8f} mAh to compute rates " \
                  "-> the mean of the 5 largest capacities found in the file.".format(self.ref_cap))
                  
        cycnums = np.arange(1, 1 + self.get_ncyc())
        
        stepnums = self.formatted_df["Prot_step"].unique()
        cycnums = np.zeros(len(stepnums), dtype='int')
        nstep = len(stepnums)
        step_rates = ['N/A']*nstep
        chg_rates = []
        chg_crates = []
        dis_rates = []
        dis_crates = []
        for i in range(len(stepnums)):
            step = self.formatted_df.loc[self.formatted_df["Prot_step"] == stepnums[i]]
            cycnum = step["Cycle"].unique()
            cycnums[i] = cycnum[0]
            
            if step["Step"].unique()[0] in [1, 5]:
                chg_cur = step["Current"].values
                chg_cur_max = np.amax(np.absolute(chg_cur))
                if chg_cur_max > 0.0:
                    cr = chg_cur_max / self.ref_cap
                    ind = np.argmin(np.absolute(RATES - cr))
                    chgrate = RATES[ind]
                    if chgrate not in chg_rates:
                        chg_rates.append(chgrate)
                        chg_crates.append(C_RATES[ind])
                    step_rates[i] = C_RATES[ind]
                
            if step["Step"].unique()[0] in [2, 6]:
                dis_cur = step["Current"].values
                dis_cur_max = np.amax(np.absolute(dis_cur))
                if dis_cur_max > 0.0:
                    cr = dis_cur_max / self.ref_cap
                    ind = np.argmin(np.absolute(RATES - cr))
                    disrate = RATES[ind]
                    if disrate not in dis_rates:
                        dis_rates.append(disrate)
                        dis_crates.append(C_RATES[ind])
                    step_rates[i] = C_RATES[ind]
                
        self.step_df = pd.DataFrame()
        self.step_df["Cycle"] = cycnums
        self.step_df["Prot_step"] = stepnums
        self.step_df["C_rates"] = step_rates
        print('Found charge C-rates: {}'.format(chg_crates))
        print('Found discharge C-rates: {}'.format(dis_crates))
        self.chg_crates = chg_crates
        self.dis_crates = dis_crates
        # Get step types for each prot_step
        #ds = self.formatted_df["Step"].ne(self.formatted_df["Step"].shift())
        ds = self.formatted_df["Prot_step"].ne(self.formatted_df["Prot_step"].shift())
        self.step_df["Step"] = self.formatted_df.loc[ds]["Step"].values


    def get_ncyc(self):
        '''
        Returns the total number of cycles.
        '''
        
        return int(self.formatted_df['Cycle'].values[-1])
    
    def get_cycnums(self):
        
        return self.formatted_df['Cycle'].values
    

    def get_rates(self, cyctype='cycle'):
        '''
        Return unique C-rates for cyctype.
        cyctype: {'cycle', 'charge', 'discharge'}
        '''
        
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
        cyctype: {'cycle', 'cyc', 'charge', ' chg', 'discharge', 'dis'}
        '''
        
        if rate not in C_RATES:
            raise ValueError('rate must be one of {0}'.format(C_RATES))

        if cyctype not in CYC_TYPES:
            raise ValueError('cyctype must be one of {0}'.format(CYC_TYPES))

        if cyctype in ['cycle', 'cyc']:
            selected_cycs = []
            cycles = self.step_df.loc[self.step_df["C_rates"] == rate]
            cycnums = cycles["Cycle"].unique()
            for i in range(len(cycnums)):
                steps_df = cycles.loc[cycles["Cycle"] == cycnums[i]]
                stepnums = steps_df["Prot_step"].values
    
                if len(stepnums) >= 2:
                    selected_cycs.append(cycnums[i])

        elif cyctype in ['charge', 'chg']:
            chg_df = self.step_df.loc[((self.step_df["Step"] == 1) | (self.step_df["Step"] == 5)) & (self.step_df["C_rates"] == rate)]
            selected_cycs = chg_df["Cycle"].values

        elif cyctype in ['discharge', 'dis']:
            dis_df = self.step_df.loc[((self.step_df["Step"] == 2) | (self.step_df["Step"] == 6)) & (self.step_df["C_rates"] == rate)]
            selected_cycs = dis_df["Cycle"].values

        #print(selected_cycs)

        return selected_cycs
    
    
    def get_discap(self, x_var='cycnum', rate=None, cyctype='cycle', 
                   normcyc=None, specific=False, vrange=None):
        '''
        Return discharge capacity 
        x_var: {'cycnum', 'time'}
        TODO: implement specific capacity. Need to add mass arg in __init__
        '''
        

        if rate is not None:
            selected_cycs = self.select_by_rate(rate, cyctype=cyctype)
        else:
            selected_cycs = self.get_cycnums()
        ncycs = len(selected_cycs)
            
        
        caps  = np.zeros(ncycs)
        x = np.zeros(ncycs)
        for i in range(ncycs):
            cyc_df = self.formatted_df.loc[self.formatted_df['Cycle'] == selected_cycs[i]]

            cap = cyc_df['Capacity'].values
            if vrange is not None:
                q, v = self.get_vcurve(cycnum=selected_cycs[i], cyctype='discharge')
                #new_df = cyc_df.loc[(cyc_df['Potential'] > vrange[0]) & (cyc_df['Potential'] < vrange[1])]
                
                inds = np.where((v > vrange[0]) & (v < vrange[1]))[0]
                if len(inds) < 2:
                    continue
                cap = q[inds]
                #cap = new_df['Capacity'].values
                #if len(cap) < 2:
                #    continue
                
            caps[i] = np.absolute(np.amax(cap) - np.amin(cap))
            if x_var == 'time':
                time = cyc_df['Time'].values
                x[i] = time[-1]
            else:
                x[i] = selected_cycs[i]
                
        if normcyc is not None:
            caps = caps / caps[normcyc - 1]
        
        return x, caps
    
    
    def get_vcurve(self, cycnum=-1, cyctype='cycle', active_mass=None):


        if cyctype not in CYC_TYPES:
            raise ValueError('cyctype must be one of {0}'.format(CYC_TYPES))

        if cycnum == -1:
            cycnum = self.get_ncyc() - 1

        try:
            cycle = self.formatted_df.loc[self.formatted_df['Cycle'] == cycnum]
        except:
            print('Cycle {} does not exist. Input a different cycle number.'.format(cycnum))
            

        if cyctype in ['charge', 'chg']:
            chg = cycle.loc[(cycle['Step'] == 1) | (cycle['Step'] == 5)]
            chg_steps = chg["Prot_step"].unique()
            nchg_steps = len(chg_steps)
            #print(nchg_steps)
            
            if len(chg) != 0:
                if self.cap_type == "cross":
                    chgstep = chg.loc[chg["Prot_step"] == chg_steps[0]]
                    Vchg = chgstep["Potential"].values
                    Cchg = chgstep["Capacity"].values
                    if nchg_steps > 1:
                        for i in range(nchg_steps - 1):
                            chgstep = chg.loc[chg["Prot_step"] == chg_steps[i+1]]
                            Vchg = np.concatenate((Vchg, chgstep["Potential"].values))
                            Cchg = np.concatenate((Cchg, Cchg[-1] + chgstep["Capacity"].values))
                            
                #voltage = chg['Potential'].values
                #capacity = chg['Capacity'].values
                return Cchg, Vchg
            
            else:
                return None, None

        elif cyctype in ['discharge', 'dis']:
            dis = cycle.loc[(cycle['Step'] == 2) | (cycle['Step'] == 6)]
            dis_steps = dis["Prot_step"].unique()
            ndis_steps = len(dis_steps)
            #print(ndis_steps)
            
            if len(dis) != 0:
                if self.cap_type == "cross":
                    disstep = dis.loc[dis["Prot_step"] == dis_steps[0]]
                    Vdis = disstep["Potential"].values
                    Cdis = disstep["Capacity"].values
                    if ndis_steps > 1:
                        for i in range(ndis_steps - 1):
                            disstep = dis.loc[dis["Prot_step"] == dis_steps[i+1]]
                            Vdis = np.concatenate((Vdis, disstep["Potential"].values))
                            Cdis = np.concatenate((Cdis, Cdis[-1] + disstep["Capacity"].values))
                            
                #voltage = dis['Potential'].values
                #capacity = dis['Capacity'].values
                return Cdis, Vdis
            else:
                return None, None

        elif cyctype in ['cycle', 'cyc']:

            chg = cycle.loc[(cycle['Step'] == 1) | (cycle['Step'] == 5)]
            #print(len(chg))
            chg_steps = chg["Prot_step"].unique()
            nchg_steps = len(chg_steps)
            #print(nchg_steps)

            if nchg_steps > 0:
                
                if self.cap_type == 'cum':
                    Vchg = chg['Potential'].values
                    Cchg = chg['Capacity'].values
                    dis = cycle.loc[(cycle['Step'] == 2) | (cycle['Step'] == 6)]
    
                    Vdis = dis['Potential'].values
                    Cdis = dis['Capacity'].values
    
                    voltage = np.concatenate((Vchg, Vdis))
                    
                    capacity = np.concatenate((Cchg, Cdis))
                    
                elif self.cap_type == "cross":
                    chgstep = chg.loc[chg["Prot_step"] == chg_steps[0]]
                    Vchg = chgstep["Potential"].values
                    Cchg = chgstep["Capacity"].values
                    if nchg_steps > 1:
                        for i in range(nchg_steps - 1):
                            chgstep = chg.loc[chg["Prot_step"] == chg_steps[i+1]]
                            Vchg = np.concatenate((Vchg, chgstep["Potential"].values))
                            Cchg = np.concatenate((Cchg, Cchg[-1] + chgstep["Capacity"].values))
                    
                    dis = cycle.loc[(cycle['Step'] == 2) | (cycle['Step'] == 6)]
                    dis_steps = dis["Prot_step"].unique()
                    ndis_steps = len(dis_steps)
                    #print(ndis_steps)
                    disstep = dis.loc[dis["Prot_step"] == dis_steps[0]]
                    Vdis = disstep["Potential"].values
                    Cdis = disstep["Capacity"].values
                    
                    if ndis_steps > 1:
                        for i in range(ndis_steps - 1):
                            disstep = dis.loc[dis["Prot_step"] == dis_steps[i+1]]
                            Vdis = np.concatenate((Vdis, disstep["Potential"].values))
                            Cdis = np.concatenate((Cdis, Cdis[-1] + disstep["Capacity"].values))
                            
                    capacity = np.concatenate((Cchg, -Cdis+Cchg[-1]))
                    voltage = np.concatenate((Vchg, Vdis))

            else:
                # May want to return discharge curve even if no charge curve exists for selected cycle number.
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

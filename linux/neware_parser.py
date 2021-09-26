#!/usr/bin/env python

import numpy as np
import pandas as pd
from os import path
from scipy.integrate import simps
import re
from argparse import ArgumentParser
from datetime import datetime


CYC_TYPES = {'charge', 'discharge', 'cycle'}
RATES = np.array([1/160, 1/80, 1/40, 1/20, 1/10, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5])
C_RATES = ['C/160', 'C/80', 'C/40', 'C/20', 'C/10', 'C/5', 'C/4', 'C/3', 'C/2', '1C', '2C', '3C', '4C', '5C']

class ParseNeware():

    def __init__(self, newarefile, all_lines=None, ref_cap=None):
        '''parse raw neware datafile into cycle, step, and record data\
           and put into pd df'''

        if all_lines is not None:
            lines = all_lines
        else:
            self.newarefile = newarefile[:-4]
            with open(newarefile, 'r', encoding='unicode_escape') as f:
                lines = f.readlines()
    
        # Replace single space between words with "_" to create column labels.
        cyclabels = re.sub(r'(\w+) (\w+)', r'\1_\2', lines[0])
        cyclabels = re.sub(r'(\w+) (\w+)', r'\1_\2', cyclabels)
        cyclabels = re.sub(r' ', r'', cyclabels)
        clabels = cyclabels.strip().split()

        steplabels = re.sub(r'(\w+) (\w+)', r'\1_\2', lines[1])
        steplabels = re.sub(r'(\w+) (\w+)', r'\1_\2', steplabels)
        steplabels = re.sub(r' ', r'', steplabels)
        slabels = steplabels.strip().split()

        reclabels = re.sub(r'(\w+) (\w+)', r'\1_\2', lines[2])
        reclabels = re.sub(r'(\w+) (\w+)', r'\1_\2', reclabels)
        reclabels = re.sub(r' ', r'', reclabels)
        rlabels = reclabels.strip().split()

        cyclnlen = len(clabels)
#        print('Found {} cycle labels.'.format(cyclnlen))
        steplnlen = len(slabels)
#        print('Found {} step labels.'.format(steplnlen))
        reclnlen = len(rlabels)
#        print('Found {} record labels.'.format(reclnlen))

        # Parse out units from column labels and create dictionary of units
        # for cycle, step, and record data.

        self.cycunits = dict()
        newclabels = []
        cycheader = ''
        for l in clabels:
            try:
                m = re.search(r'\(.*\)', l)
                newlab = l[:m.start()]
                if newlab == 'Specific_Capacity-Dchg': 
                    newlab = 'Specific_Capacity-DChg' 
                if newlab == 'RCap_Chg':
                    newlab = 'Specific_Capacity-Chg'
                if newlab == 'RCap_DChg':
                    newlab = 'Specific_Capacity-DChg'
                self.cycunits[newlab] = l[m.start()+1:m.end()-1]
                newclabels.append(newlab)
                cycheader = cycheader + '\t{}'.format(newlab)
            except:
                self.cycunits[l] = None
                newclabels.append(l)
                cycheader = cycheader + '\t{}'.format(l)

        self.stepunits = dict()
        stepheader = 'Cycle_ID'
        newslabels = ['Cycle_ID']
        for l in slabels:
            try:
                m = re.search(r'\(.*\)', l)
                newlab = l[:m.start()]
                self.stepunits[newlab] = l[m.start()+1:m.end()-1]
                newslabels.append(newlab)
                stepheader = stepheader + '\t{}'.format(newlab)
            except:
                self.stepunits[l] = None
                if l == 'Step_Name':
                    l = 'Step_Type'
                newslabels.append(l)
                stepheader = stepheader + '\t{}'.format(l)

        self.recunits = dict()
        recheader = 'Cycle_ID\tStep_ID'
        newrlabels = ['Cycle_ID', 'Step_ID']
        for l in rlabels:
            try:
                m = re.search(r'\(.*\)', l)
                newlab = l[:m.start()]
                if newlab == 'Vol':
                    newlab = 'Voltage'
                if newlab == 'Cap':
                    newlab = 'Capacity'
                if newlab == 'CmpCap':
                    newlab = 'Capacity_Density'
                if newlab == 'Cur':
                    newlab = 'Current'
                self.recunits[newlab] = l[m.start()+1:m.end()-1]
                newrlabels.append(newlab)
                recheader = recheader + '\t{}'.format(newlab)
            except:
                self.recunits[l] = None
                newrlabels.append(l)
                recheader = recheader + '\t{}'.format(l)

        # Create header line for cycle, step, and record data
        cyc = ['{}\n'.format(cycheader)]
        step = ['{}\n'.format(stepheader)]
        rec = ['{}\n'.format(recheader)]

        # Separate cycle, step, and record data and write to 
        # file (needs to be changed to tmpfile) to be read 
        # in as DataFrame with inferred dtypes.
        cyc_nlws = len(lines[0]) - len(lines[0].lstrip())
        step_nlws = len(lines[1]) - len(lines[1].lstrip())
        rec_nlws = len(lines[2]) - len(lines[2].lstrip())
        for line in lines[3:]:
            l = line.strip().split()
            nlws = len(line) - len(line.lstrip())
            if nlws == cyc_nlws:
                cycnum = l[0]
                cyc.append(line)

            elif nlws == step_nlws: 
                stepnum = l[0]
                step.append('{0}{1}'.format(cycnum, line))

            else:
                rec.append('{0}\t{1}{2}'.format(cycnum, stepnum, line[1:]))

#            if len(l) == cyclnlen:
#                cycnum = l[0]
#                cyc.append(line)
#            elif len(l) == steplnlen:
#                stepnum = l[0]
#                step.append('{0}{1}'.format(cycnum, line))
#            else:
#                rec.append('{0}\t{1}{2}'.format(cycnum, stepnum, line))
        # Temporary fix: path variable for writing cyc, step, rec to file.
        tmp_path = '.'
        
        with open('{}/cyc.dat'.format(tmp_path), 'w') as f:
            for l in cyc:
                f.write(l)
        with open('{}/step.dat'.format(tmp_path), 'w') as f:
            for l in step:
                f.write(l)
        with open('{}/rec.dat'.format(tmp_path), 'w') as f:
            for l in rec:
                f.write(l)

        self.cyc = pd.read_csv('{}/cyc.dat'.format(tmp_path), sep='\t+', header=0, engine='python')
        self.step = pd.read_csv('{}/step.dat'.format(tmp_path), sep='\t+', header=0, engine='python')
        self.rec = pd.read_csv('{}/rec.dat'.format(tmp_path), sep='\t+', header=0, engine='python')
        
        if self.recunits['Voltage'] == 'mV':
            self.rec['Voltage'] = self.rec['Voltage'] / 1000
            self.recunits['Voltage'] = 'V'
        ## =============================================== ##

        # Need codes checked
        step_type = {
            "rest" : 0,
            "cc_dchg" : 2,
            "cc_chg" : 1,
            "cccv_chg" : 5,
            "cccv_dchg" : 6
        }
        
        univ_cols = ["Time", "Cycle", "Step", "Current", "Potential", "Capacity", "Prot_step"]
        universal_df = pd.DataFrame(columns=univ_cols)
        
        #t = pd.to_datetime(self.rec["Realtime"], format='%Y-%m-%d %H:%M:%S')
        t = pd.to_datetime(self.rec["Realtime"])
        delta = t - t[0]
        universal_df["Time"] = delta.dt.total_seconds()
        
        univ_prot_step = self.rec["Step_ID"].values
        if univ_prot_step[0] > 0:
            univ_prot_step = univ_prot_step - univ_prot_step[0]
        universal_df["Prot_step"] = univ_prot_step
        
        mapped_steps = self.step["Step_Type"].str.lower().map(step_type)
        prosteps = self.step["Step_ID"].values
        if prosteps[0] > 0:
            prosteps = prosteps - prosteps[0]
        tmp_df = pd.DataFrame(data={"Step": mapped_steps, "Prot_step": prosteps})
        universal_df["Step"] = universal_df["Prot_step"].map(tmp_df.set_index("Prot_step")["Step"])
        
        # If first cycle does not contain a charge, set to cycle 0. Otherwise cycle 1.
        universal_df["Cycle"] = self.rec["Cycle_ID"]
        first_cyc = universal_df.loc[universal_df["Cycle"] == 1]
        steps = first_cyc["Step"].unique()
        if (1 not in steps) and (5 not in steps):
            if universal_df["Cycle"][0] > 0:
                universal_df["Cycle"] = universal_df["Cycle"] - 1
        
        universal_df["Current"] = self.rec["Current"]
        universal_df["Potential"] = self.rec["Voltage"]
        universal_df["Capacity"] = self.rec["Capacity"]
        self.cap_type = "cross"
        
        '''
        # Convert capacity to cumulative.
        nrec = len(self.rec["Capacity"])
        cap = np.zeros(nrec, dtype="float")
        cycnums = universal_df["Cycle"].values
        ind = 0
        ref_cap = 0.0
        if cycnums[0] == 0:
            start_ind = 1
            first_cyc = universal_df.loc[universal_df["Cycle"] == 0]
            prosteps = first_cyc["Prot_step"].unique()
            for j in range(len(prosteps)):
                tmp_cap = first_cyc.loc[first_cyc["Prot_step"] == prosteps[j]]["Capacity"].values
                n = len(tmp_cap)
                if prosteps[j] not in [2, 6]:     
                    cap[ind:ind+n] = tmp_cap
                    #ref_cap = tmp_cap[-1]
                else:
                    cap[ind:ind+n] = tmp_cap[-1] - tmp_cap
                    ref_cap = cap[ind+n-1]
                ind = ind + n
        else:
            start_ind = 0
        
        for i in range(start_ind, len(cycnums)):
            cyc_df = universal_df.loc[universal_df["Cycle"] == cycnums[i]]
            steps = cyc_df["Step"].unique()
            for j in range(len(steps)):
                tmp_cap = cyc_df.loc[cyc_df["Step"] == prosteps[j]]["Capacity"].values
                n = len(tmp_cap)
                try:
                    if prosteps[j] in [1, 5]:
                        cap[ind:ind+n] = ref_cap + tmp_cap
                    elif prosteps[j] in [2, 6]:
                        cap[ind:ind+n] = ref_cap - tmp_cap
                    elif prosteps[j] not in [1, 2, 5, 6]:
                    #else:
                        cap[ind:ind+n] = tmp_cap
                except:
                    print(j, cycnums[i], prosteps[j])
                    
                    raise SystemExit
                ref_cap = cap[ind+n-1]
                ind = ind + n

        universal_df["Capacity"] = cap
        '''
        self.universal_df = universal_df
        
        ### What is commented below can be removed by John. 
        ### Just left for his reference.
        '''
        t_i = datetime.strptime(self.rec["Realtime"][0], '%Y-%m-%d %H:%M:%S')

        univ_t = []
        
        
        for val in self.rec['Realtime'].values.tolist():
            univ_t.append((datetime.strptime(str(val), '%Y-%m-%d %H:%M:%S') - t_i).total_seconds() / 3600)
         
        univ_cyc = self.rec["Cycle_ID"].values.tolist()
        
        univ_prot_step = np.array(self.rec["Step_ID"].values.tolist())
        if univ_prot_step[0] != 0:
            univ_prot_step -= univ_prot_step[0]
       
        
        univ_step_code = []
        
        for val in univ_prot_step:
            univ_step_code.append(step_type[self.step["Step_Type"][val].lower()])
#            univ_step_code.append(step_type[self.step["Step_Name"][val].lower()])
            
        univ_curr = []
            
        for val in self.rec["Current"].values.tolist():
            univ_curr.append(val)
            
        univ_v = []
            
        for val in self.rec["Voltage"].values.tolist():
            univ_v.append(val)
            
        univ_q = []
        
        for val in self.rec["Capacity"].values.tolist():
            univ_q.append(val)
            
        univ_cd = []
        
        for val in self.rec["Capacity_Density"].values.tolist():
            univ_cd.append(val)

        
        univ_df_rows = []
        univ_cols = ["Time", "Cycle", "Step", "Current", "Potential", "Capacity", "Prot_step", "Capacity_Density"]
        
        for i in range(len(univ_q)):
            univ_df_rows.append([univ_t[i], univ_cyc[i], univ_step_code[i], univ_curr[i], univ_v[i], univ_q[i], univ_prot_step[i], univ_cd[i]])
                     
        universal_format = pd.DataFrame(univ_df_rows, columns=univ_cols)
        
        self.universal_format = universal_format
            
        universal_format.to_csv('univ_format.csv', index=False)
        '''
        ## =============================================== ##

        # Convert Capacity_Density to mAh/g from mAh/kg. Unit label can be wrong.
        cyc2_df = self.rec.loc[self.rec['Cycle_ID'] == 3]
        max_cap = cyc2_df['Capacity_Density'].values
        if np.amax(max_cap) > 1000:
            self.rec['Capacity_Density'] = self.rec['Capacity_Density'] / 1000
            self.cyc['Specific_Capacity-DChg'] = self.cyc['Specific_Capacity-DChg'] / 1000  
        if self.recunits['Capacity_Density'] == 'mAh/kg':
            self.recunits['Capacity_Density'] = 'mAh/g'
            
    def get_rec(self):
        return self.rec
            
    def get_universal_format(self):
        return self.universal_df
            
            
    # The following set of functions are designed to be intuitive 
    # for people in the lab that want to get particular information
    # from cycling data.

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
        

    def get_ncyc(self):
        '''
        Returns the total number of cycles.
        '''
        return int(self.cyc['Cycle_ID'].values[-1])

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
        stepdata = self.step.loc[self.step['C_rate'] == rate]
        cycnums = stepdata['Cycle_ID'].unique()
        for i in range(len(cycnums)):
            stepnums = stepdata.loc[stepdata['Cycle_ID'] == cycnums[i]].values
            if cyctype == 'cycle':
                if len(stepnums) >= 2:
                    selected_cycs.append(cycnums[i])

            elif cyctype == 'charge':
                step = self.step.loc[self.step['Step_ID'] == stepnums[0]]
                if step['C_rate'].values == rate:
                    selected_cycs.append(cycnums[i])

            elif cyctype == 'discharge':
                step = self.step.loc[self.step['Step_ID'] == stepnums[-1]]
                if step['C_rate'].values == rate:
                    selected_cycs.append(cycnums[i])

# For cyctype='cycle' need to check that first and last step both have same rate.
# For charge/discharge need to check that first/last step are at C_rate=rate

        return selected_cycs

    def get_discap(self, normcyc=None, specific=True):
        '''
        Returns the cycle numbers and discharge capacity
        '''
        if specific is True:
            caplabel = 'Specific_Capacity-DChg'
        else:
            caplabel = 'Cap_DChg'

        if normcyc is not None:
            normcyc = int(normcyc)
            cap = self.cyc[caplabel]
            return self.cyc['Cycle_ID'], cap / cap[normcyc]

        else:
            return self.cyc['Cycle_ID'], self.cyc[caplabel]

    def get_chgcap(self, normcyc=None, specific=True):
        '''
        Returns the cycle numbers and charge capacity
        '''
        if specific is True:
            caplabel = 'Specific_Capacity-Chg'
        else:
            caplabel = 'Cap_Chg'

        if normcyc is not None:
            normcyc = int(normcyc)
            cap = self.cyc[caplabel]
            return self.cyc['Cycle_ID'], cap / cap[normcyc]

        else:
            return self.cyc['Cycle_ID'], self.cyc[caplabel]

    def get_deltaV(self, normcyc=None, cycnums=None):
        '''
        Return the difference between average charge and discharge voltages
        computed by average value theorem (integrate V-Q)
        NOTE: This still needs work. Crashes if V-Q curve data is too noisy. 
        '''
        if cycnums is not None:
            cycle_nums = cycnums
        else:
            cycle_nums = np.arange(1, self.get_ncyc() + 1)

#        dVfile = '{}-deltaV.csv'.format(self.newarefile)
#        if path.isfile(dVfile) is True:
#            df = pd.read_csv(dVfile)
#            cycnums = df['Cycle_ID'].values
#            dV = df['Delta_V'].values

#        else:
#            if normcyc is not None:
#                startcyc = normcyc
        bad_inds = []
#        cycnums = np.arange(1, self.get_ncyc()+1)
        dV = np.zeros(len(cycle_nums), dtype=float)
        for i in range(len(cycle_nums)):
            Qchg, Vchg = self.get_vcurve(cycnum=i+1, cyctype='charge')
            Qdis, Vdis = self.get_vcurve(cycnum=i+1, cyctype='discharge')
            if ( (Qchg[-1] - Qchg[0]) < 0.001 ) or ( (Qdis[-1] - Qdis[0]) < 0.001 ):
#                    bad_inds.append(i)
                continue
            else:
                dVchg = (1/(Qchg[-1] - Qchg[0]))*simps(Vchg, Qchg)
                dVdis = (1/(Qdis[-1] - Qdis[0]))*simps(Vdis, Qdis)
            
                dV[i] = dVchg - dVdis

        if len(bad_inds) > 0:
            dV = np.delete(dV, bad_inds)
            cycle_nums = np.delete(cycle_nums, bad_inds)

#        df = pd.DataFrame(data={'Cycle_ID': cycnums, 'Delta_V': dV})
#        df.to_csv(path_or_buf=dVfile, index=False)
#        print(bad_inds)
        
        if normcyc is not None:
            return cycle_nums, dV/dV[normcyc]

        else:
            return cycle_nums, dV

    def get_vcurve(self, cycnum=-1, cyctype='cycle'):
        '''
        Get voltage curve (cap, V) for a specific cycle number (cycnum).
        TODO: Need to deal with error handling properly.
        '''
        if cyctype not in CYC_TYPES:
            raise ValueError('cyctype must be one of {0}'.format(CYC_TYPES))

        if cycnum == -1:
            cycnum = self.get_ncyc() - 1

        try:
            cycle = self.rec.loc[self.rec['Cycle_ID'] == cycnum]
        except:
#            raise Exception('Cycle {} does not exist. Input a different cycle number.'.format(cycnum))
            print('Cycle {} does not exist. Input a different cycle number.'.format(cycnum))

        stepnums = cycle['Step_ID'].unique()
        
        if cyctype == 'charge':
            chg = cycle.loc[cycle['Step_ID'] == stepnums[0]]
            voltage = chg['Voltage'].values
            capacity = chg['Capacity_Density'].values

        elif cyctype == 'discharge':
            dis = cycle.loc[cycle['Step_ID'] == stepnums[-1]]
            voltage = dis['Voltage'].values
            capacity = dis['Capacity_Density'].values
            
        elif cyctype == 'cycle':
            chg = cycle.loc[cycle['Step_ID'] == stepnums[0]]
            Vchg = chg['Voltage'].values
            Cchg = chg['Capacity_Density'].values

            dis = cycle.loc[cycle['Step_ID'] == stepnums[-1]]
            Vdchg = dis['Voltage'].values
            Cdchg = dis['Capacity_Density'].values

            voltage = np.concatenate((Vchg, Vdchg))
            capacity = np.concatenate((Cchg, -Cdchg+Cchg[-1]))

        return capacity, voltage


    def get_dQdV(self, cycnum=-1, cyctype='cycle', avgstride=None):
        '''
        Get dQdV for specific cycle. Returns charge and discharge together.
        TODO: Add running average. 
        '''
        cchg, vchg = self.get_vcurve(cycnum=cycnum, cyctype='charge')
        cend = cchg[-1]
 #       vchg, cchg = vchg[:-1], cchg[:-1]
        delta_cchg = cchg[1:] - cchg[:-1]  
        delta_vchg = vchg[1:] - vchg[:-1]
        inf_inds = np.where(np.absolute(delta_vchg) < 1e-12)
        vchg = np.delete(vchg, inf_inds[0] + 1)
        cchg = np.delete(cchg, inf_inds[0] + 1)
        dQdVchg = (cchg[1:] - cchg[:-1]) / (vchg[1:] - vchg[:-1])
#        vchg = (vchg[1:] + vchg[:-1]) / 2

        cdchg, vdchg = self.get_vcurve(cycnum=cycnum, cyctype='discharge')
        cdchg = -cdchg + cend
#        vdchg, cdchg = vdchg[1:], cdchg[1:]
        delta_cdchg = cdchg[1:] - cdchg[:-1]  
        delta_vdchg = vdchg[1:] - vdchg[:-1]
        inf_inds = np.where(np.absolute(delta_vdchg) < 1e-12)
        vdchg = np.delete(vdchg, inf_inds[0] + 1)
        cdchg = np.delete(cdchg, inf_inds[0] + 1)
        dQdVdchg = -(cdchg[1:] - cdchg[:-1]) / (vdchg[1:] - vdchg[:-1])
#       vdchg = (vdchg[1:] + vdchg[:-1]) / 2

        voltage = np.concatenate((vchg[1:], vdchg[:-1])) 
        dQdV = np.concatenate((dQdVchg, dQdVdchg))
 
        if avgstride is not None:
            voltage, dQdV = self.runavg(voltage, dQdV, avgstride)

        return voltage, dQdV
#        return np.concatenate((vchg[1:], vdchg[:-1])), np.concatenate((dQdVchg, dQdVdchg))

    def runavg(self, xarr, yarr, avgstride):
        '''
        Running average.
        '''
        window = avgstride*2 + 1
        weights = np.repeat(1.0, window)/window
        avgdata = np.convolve(yarr, weights, 'valid')

        return xarr[avgstride: -avgstride], avgdata


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('newarefile')
    args = parser.parse_args()

    # this does nothing currently. Need to think about 
    # what command line utils to include.
    nd = ParseNeware(args.newarefile)



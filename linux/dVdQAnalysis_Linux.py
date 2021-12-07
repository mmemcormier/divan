"""
Title: Neware Analysis Tool
Created on Monday, May 10, 2021
@author: John Corsten
"""

import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
from bokeh.layouts import row, column, gridplot
from bokeh.models import Legend, LegendItem
from bokeh.plotting import figure, save
from universal_format import UniversalFormat
from fractions import Fraction
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import bokeh.palettes as bp
from scipy.signal import savgol_filter
## Windows only
from pathlib import Path
from bokeh.io import export_png
import tkinter as tk
from tkinter import filedialog
import io
import pandas as pd


'''
## dV/dQ Analysis
'''

# File selection widget
file_expander = st.expander("Load files here")
# Expander can be opened or closed using the +/- button to hide the data selection widget
with file_expander:
    fullData = st.file_uploader("Load Neware data file file here!")
    st.markdown('''
    #### If you wish to perform dV/dQ analysis, select positive and negative reference files:
    ''')
    posData = st.file_uploader("Load the positive reference file here!")
    negData = st.file_uploader("Load the negative reference file here!")
    reference_type = st.radio("Fitting full cell charge or discharge?", ('Discharge', 'Charge'))

    # ============================================================================ #
    # Windows only feature

    # Set up tkinter
    root = tk.Tk()
    root.withdraw()

    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)
    # ============================================================================ #


    st.write("When you are finished selecting files, click the '-' button at the "
             "top right of this widget to minimize it.")

# Reading in Neware data, and caching data
@st.cache(persist=True, show_spinner=True)
def read_data(uploaded_bytes, cell_id):
    if uploaded_bytes.type == 'application/vnd.ms-excel':
        file = 'csv'
        buffer = (uploaded_bytes.getbuffer())
        data = buffer.tobytes().decode(encoding="utf-8").splitlines()
    else:
        file = "txt"
        data = uploaded_bytes
        
    uf = UniversalFormat(file, all_lines=data)
    return uf

# Reading uploaded reference files, and caching data
@st.cache(persist=True, show_spinner=False)
def read_ref(pData, nData):
    
    v_n, q_n, v_p, q_p = [], [], [], []
    plines = []
    nlines = []

    try:
        v_p, q_p = np.loadtxt(pData, skiprows=1, unpack=True)

    except:
        for line in pData:
            plines.append(line[:-2].decode("utf-8"))
        pData_df = pd.DataFrame([r.split(',') for r in plines][1:])

        v_p = pData_df[0].astype(np.float)
        q_p = pData_df[1].astype(np.float)


    try:
        v_n, q_n = np.loadtxt(nData, skiprows=1, unpack=True)
        
    except:
        for line in nData:
            nlines.append(line[:-2].decode("utf-8"))

        nData_df = pd.DataFrame([r.split(',') for r in nlines][1:])
        v_n = nData_df[0].astype(np.float)
        q_n = nData_df[1].astype(np.float)


    return np.array(v_n), np.array(q_n), np.array(v_p), np.array(q_p)

@st.cache(persist=True, show_spinner=False)
def voltage_curves(cycnums, cyctype="charge", active_mass=None):
    cap_list = []
    volt_list = []
    for cycnum in cycnums:
        cap, volt = nd.get_vcurve(cycnum=cycnum, cyctype=cyctype)
        cap_list.append(cap)
        volt_list.append(volt)

    return cap_list, volt_list

@st.cache(persist=True, show_spinner=False)
def dqdv_curves(cycnums, active_mass=None, explore=False):
    V_list = []
    dqdv_list = []
    for cycnum in cycnums:
        volt, dqdv = nd.get_dQdV(cycnum=cycnum)

        if explore is False:
            inds = np.argmax(volt)
            if reference_type == "Discharge":
                volt = volt[:inds]
                dqdv = dqdv[:inds]
            else:
                volt = volt[inds:]
                dqdv = dqdv[inds:]

        V_list.append(volt)
        dqdv_list.append(dqdv)

    return V_list, dqdv_list

@st.cache(persist=True, show_spinner=False)
def dVdQ_m(capacity_m, voltage_m, active_mass=None):
    # Get discharge part of cycle only.
    inds = np.argmax(capacity_m)
    if reference_type == "Discharge":
        capacity_m = capacity_m[:inds]
        voltage_m = voltage_m[:inds]
    else:
        capacity_m = capacity_m[inds:]
        voltage_m = voltage_m[inds:]

    dvolt = voltage_m[1:] - voltage_m[:-1]
    dcap = capacity_m[1:] - capacity_m[:-1]
    bad_inds = np.where(np.absolute(dcap) < 0.0001)[0]
    dcap = np.delete(dcap, bad_inds)
    dvolt = np.delete(dvolt, bad_inds)

    new_cap = (capacity_m[1:] + capacity_m[:-1]) / 2
    new_cap = np.delete(new_cap, bad_inds)

    return new_cap, (dvolt / dcap)

@st.cache(persist=True, show_spinner=False)
def dQdV_m(capacity_m, voltage_m, active_mass=None):
    # Get discharge part of cycle only.
    inds = np.argmax(capacity_m)
    if reference_type == "Discharge":
        capacity_m = capacity_m[:inds]
        voltage_m = voltage_m[:inds]
    else:
        capacity_m = capacity_m[inds:]
        voltage_m = voltage_m[inds:]

    dvolt = voltage_m[1:] - voltage_m[:-1]
    dcap = capacity_m[1:] - capacity_m[:-1]
    bad_inds = np.where(np.absolute(dcap) < 0.0001)[0]
    dcap = np.delete(dcap, bad_inds)
    dvolt = np.delete(dvolt, bad_inds)

    new_volt = (volt_m[1:] + volt_m[:-1]) / 2
    new_volt = np.delete(new_volt, bad_inds)

    return new_volt, (dcap / dvolt)

def monoton_check(x, y):
        dx = x[1:] - x[:-1]
        inds = np.where(dx <= 0.0)[0]
        x_new = np.delete(x,list(inds+1))
        y_new = np.delete(y,list(inds+1))
        
        return x_new, y_new

#@st.cache(persist=True, show_spinner=False)
def dVdQ_c(pos_slip, neg_slip, pos_mass, neg_mass):
    Q_p = q_p * pos_mass + pos_slip
    Q_n = q_n * neg_mass + neg_slip
    Qarr = np.arange(max(Q_p[0], Q_n[0]), min(Q_p[-1], Q_n[-1]), 0.1)

    #Q_p, v_p = monoton_check(Q_p, v_p)
    #Q_n, v_n = monoton_check(Q_n, v_n)

    Q_p_int = UnivariateSpline(Q_p, v_p, s=st.session_state['s_pos'])
    V_p_int = Q_p_int(Qarr)
    Q_n_int = UnivariateSpline(Q_n, v_n, s=st.session_state['s_neg'])
    V_n_int = Q_n_int(Qarr)

    dVdQ = np.diff(V_p_int) / np.diff(Qarr) - np.diff(V_n_int) / np.diff(Qarr)
    Qarr = (Qarr[1:] + Qarr[:-1]) / 2

    return Qarr, dVdQ

@st.cache(persist=True, show_spinner=False)
def reference_dVdQ_c(ref_type, s_neg, s_pos):
    
    if ref_type not in ['Positive Reference','Negative Reference']:
        return None, None
    
    Q_p = q_p * st.session_state['m_pos'] + st.session_state['slip_pos']
    Q_n = q_n * st.session_state['m_neg'] + st.session_state['slip_neg']
    Qarr = np.arange(max(Q_p[0], Q_n[0]), min(Q_p[-1], Q_n[-1]), 0.1)

    Qarr = (Qarr[1:] + Qarr[:-1]) / 2

    if ref_type == 'Positive Reference':
        Q_p_int = UnivariateSpline(Q_p, v_p, s=s_pos)
        V_p_int = Q_p_int(Qarr)
        
        return Qarr[1:], np.diff(V_p_int) / np.diff(Qarr)
    
    elif ref_type == 'Negative Reference':
        Q_n_int = UnivariateSpline(Q_n, v_n, s=s_neg)
        V_n_int = Q_n_int(Qarr)
        return Qarr[1:], np.diff(V_n_int) / np.diff(Qarr)
    

    return Qarr, dVdQ

#@st.cache(persist=True, show_spinner=False)
def dQdV_c(pos_slip, neg_slip, pos_mass, neg_mass):
    Q_p = q_p * pos_mass + pos_slip
    Q_n = q_n * neg_mass + neg_slip
    Qarr = np.arange(max(Q_p[0], Q_n[0]), min(Q_p[-1], Q_n[-1]), 0.1)

    #Q_p, v_p = monoton_check(Q_p, v_p)
    #Q_n, v_pn = monoton_check(Q_n, v_n)

    Q_p_int = UnivariateSpline(Q_p, v_p, s=0)
    V_p_int = Q_p_int(Qarr)
    Q_n_int = UnivariateSpline(Q_n, v_n, s=0)
    V_n_int = Q_n_int(Qarr)

    full_cell_volt = V_p_int - V_n_int

    dQdV = np.diff(Qarr) / np.diff(full_cell_volt)

    return full_cell_volt, dQdV



#@st.cache(persist=True, show_spinner=False)
def dVdQ_fitting(Qvals, pos_slip, neg_slip, pos_mass, neg_mass):
    Q_p = q_p * pos_mass + pos_slip
    Q_n = q_n * neg_mass + neg_slip
    Qarr = np.arange(max(Q_p[0], Q_n[0]), min(Q_p[-1], Q_n[-1]), 0.1)

    #Q_p, v_p = monoton_check(Q_p, v_p)
    #Q_n, v_pn = monoton_check(Q_n, v_n)

    Q_p_int = UnivariateSpline(Q_p, v_p, s=st.session_state['s_pos'])
    V_p_int = Q_p_int(Qarr)
    Q_n_int = UnivariateSpline(Q_n, v_n, s=st.session_state['s_neg'])
    V_n_int = Q_n_int(Qarr)

    dVdQ = np.diff(V_p_int) / np.diff(Qarr) - np.diff(V_n_int) / np.diff(Qarr)
    Qarr = (Qarr[1:] + Qarr[:-1]) / 2

    dVdQ_calc = np.zeros(len(Qvals))
    for i in range(len(Qvals)):
        dQ = np.absolute(Qarr - Qvals[i])
        min_ind = np.argmin(dQ)
        dVdQ_calc[i] = dVdQ[min_ind]

    return dVdQ_calc

#@st.cache(persist=True, show_spinner=False)
def interpolate_reference(v_n, q_n, v_p, q_p):
    # q_range_n is the range of specific capacity present in the negative reference data, spaced by 0.01
    q_range_n = np.arange(round(min(q_n), 2), round(max(q_n), 2), 0.01)
    # Using UnivariantSpline to interpolate values of v
    
    #q_n, v_n = monoton_check(q_n, v_n)
    spline_n = UnivariateSpline(q_n, v_n, s=st.session_state['s_pos'])
    int_v_n = spline_n(q_range_n)

    # q_range_p is the range of specific capacity present in the positive reference data, spaced by 0.01
    q_range_p = np.arange(round(min(q_p), 2), round(max(q_p), 2), 0.01)
    # Again, interpolating values of v
    #q_p, v_p = monoton_check(q_p, v_p)
    spline_p = UnivariateSpline(q_p, v_p, s=0)
    int_v_p = spline_p(q_range_p)

    # Calculating dVdq for pos. and neg. ref (remember, this is with respect to specific capacity, "little q")
    dVdq_p = np.diff(int_v_p) / np.diff(q_range_p)
    dVdq_n = np.diff(int_v_n) / np.diff(q_range_n)

    return dVdq_n, dVdq_p, q_range_p, q_range_n

#@st.cache(persist=True, show_spinner=False)
def smooth_meas(dVdQ_meas, window, polyorder):
    if window < len(dVdQ_meas):
        meas_smooth = savgol_filter(dVdQ_meas, window_length=window, polyorder=polyorder, deriv=0, delta=1.0, axis=- 1,
                                          mode='interp', cval=0.0)
    else:
        meas_smooth = dVdQ_meas

    return meas_smooth

def brute_force_fit(m_p_i, m_p_min, m_p_max, m_p_int, m_n_i, m_n_min, m_n_max, m_n_int, s_p_i,
                    s_p_min, s_p_max, s_p_int, s_n_i, s_n_min, s_n_max, s_n_int, dVdQ_measured, Q_measured):

    if lock_pm:
        m_p_vals = [m_p_i]
    else:
        m_p_vals = np.arange(m_p_min, m_p_max, m_p_int)

    if lock_nm:
        m_n_vals = [m_n_i]
    else:
        m_n_vals = np.arange(m_n_min, m_n_max, m_n_int)

    if lock_ps:
        s_p_vals = [s_p_i]
    else:
        s_p_vals = np.arange(s_p_min, s_p_max, s_p_int)

    if lock_ns:
        s_n_vals = [s_n_i]
    else:
        s_n_vals = np.arange(s_n_min, s_n_max, s_n_int)

    best_X2 = np.Inf
    best_dVdQ = []
    m_n_fit = m_n_i
    m_p_fit = m_p_i
    s_n_fit = s_n_i
    s_p_fit = s_p_i

    prog_write = st.empty()
    prog_write.markdown("""Brute force progress:""")

    prog_count = 0
    iters = len(m_p_vals)
    progress = st.progress(prog_count)

    for mp in m_p_vals:
        prog_count += 1
        progress.progress(prog_count/iters)
        for mn in m_n_vals:
            for sp in s_p_vals:
                for sn in s_n_vals:
                    dVdQ = dVdQ_fitting(Q_measured, sp, sn, mp, mn)
                    X2 = sum(((dVdQ - dVdQ_measured) ** 2) / dVdQ_measured)
                    if X2 < best_X2:
                        best_X2 = X2
                        m_n_fit = mn
                        m_p_fit = mp
                        s_n_fit = sn
                        s_p_fit = sp
                        best_dVdQ = dVdQ
    prog_write.empty()
    progress.empty()


    return best_dVdQ, m_n_fit, m_p_fit, s_n_fit, s_p_fit


@st.cache(persist=True, show_spinner=False)
def getRates():
    return nd.get_rates()


@st.cache(persist=True, show_spinner=False)
def selectByRate(rate):
    if rate == 'All':
        cyc_nums = np.arange(1, ncycs + 1)
    else:
        # There can be strange issues with the data, where dVdQ_m returns an empty list for Q_meas. In these cases,.
        #   the cycle number is skipped.
        cyc_nums = []
        for c in np.array(nd.select_by_rate(rate)):
            cap_m, volt_m = nd.get_vcurve(cycnum=c)
            Q_meas, dVdQ_meas = dVdQ_m(cap_m, volt_m)

            if len(Q_meas) == 0:
                continue
            else:
                cyc_nums.append(c)

    return cyc_nums


@st.cache(persist=True, show_spinner=False)
def dVdQ_rates(cyclerRates):
    if '/' in fastest_checkup:
        checkup = Fraction(fastest_checkup.replace('C','1'))
    else:
        checkup = float(fastest_checkup.replace('C',''))
    
    for r in cyclerRates:
        if '/' in r:
            if Fraction(r.replace('C', '1')) <= checkup:
                rates.append(r)
        else:
            if float(r.replace('C','')) <= checkup:
                rates.append(r)
    return rates


#@st.cache(persist=False, show_spinner=False)
def least_squares_fit(Q_ls, dVdQ_ls, ps, ns, pm, nm, ps_min, ns_min, pm_min, nm_min, ps_max, ns_max, pm_max, nm_max):
    p0 = [ps, ns, pm, nm]
    eps = 1e-8

    if lock_ps:
        ps_min, ps_max = ps-eps, ps+eps
    if lock_ns:
        ns_min, ns_max = ns-eps, ns+eps
    if lock_pm:
        pm_min, pm_max = pm-eps, pm+eps
    if lock_nm:
        nm_min, nm_max = nm-eps, nm+eps

    bounds = ([ps_min, ns_min, pm_min, nm_min], [ps_max, ns_max, pm_max, nm_max])

    popt, pcov = curve_fit(dVdQ_fitting, Q_ls, dVdQ_ls, p0=p0, bounds=bounds, max_nfev=1000,
                           ftol=1e-5, xtol=None, gtol=None, method='dogbox')

    # Setting the session state values (slider values) to the output of curve_fit
    return round(popt[0],4), round(popt[1],4), round(popt[2],4), round(popt[3],4)


def plotting(Q_measured, dVdQ_measured, cycle_number, save_plot, display_plot):
    dVdQ_calc = dVdQ_fitting(Q_measured, st.session_state["slip_pos"], st.session_state["slip_neg"],
                             st.session_state["m_pos"], st.session_state["m_neg"])

    Q = Q_measured

    p = figure(title="Cycle {}".format(str(cycle_number)), plot_width=600, x_range=(0, 200), y_range=(0, 0.01), plot_height=400,
               x_axis_label='Capacity, Q (mAh)',
               y_axis_label='dV/dQ (V/mAh)')

    b = p.vbar(x=[(Q_measured[cap_range_first_ind] + Q_measured[cap_range_last_ind]) / 2],
               width=(Q_measured[cap_range_last_ind] -
                      Q_measured[cap_range_first_ind]),
               bottom=0, top=10, color=['grey'], alpha=0.3)

    c = p.line(Q, dVdQ_calc)
    m = p.line(Q_measured, dVdQ_measured, color="red")

    legend = Legend(items=[
        LegendItem(label="Calculated", renderers=[c], index=0),
        LegendItem(label="Measured", renderers=[m], index=1),
        LegendItem(label="Fit range", renderers=[b], index=2)
    ])

    p.add_layout(legend)

    # =================================================================#
    # Windows only feature #
    if save_plot:
        if dirname is None:
            export_png(p, filename="cycle_{}_fit.png".format(cycle_number))
        else:
            export_png(p, filename=dirname + "cycle_{}_fit.png".format(cycle_number))
    # =================================================================#


    if display_plot:
        st.bokeh_chart(p, use_container_width=True)

    return dVdQ_calc

# If a Neware file has been uploaded
if fullData is not None:

    #nd, uf_rates = read_data(fullData, "Cell_ID")
    nd = read_data(fullData, "Cell_ID")
    uf_rates = nd.get_rates()
    ncycs = nd.get_ncyc()



    # Options for what to plot
    # Only provide option of 'dV/dQ' if reference curves have been uploaded
    if posData is not None and negData is not None:
        plot_opts = st.sidebar.selectbox("What would you like to plot?",
                                         ('None', 'Cell Explorer','dV/dQ'))
        
        v_n, q_n, v_p, q_p = read_ref(posData, negData)
        q_n, v_n = monoton_check(q_n, v_n)
        q_p, v_p = monoton_check(q_p, v_p)
        
    else:
        plot_opts = st.sidebar.selectbox("What would you like to plot?",
                                         ('None', 'Cell Explorer'))

    # Selecting available cycle rates
    rates = []
    cyc_nums = []
    
    if 'm_pos' not in st.session_state:
        st.session_state["m_pos"] = 1.0

    if 'm_neg' not in st.session_state:
        st.session_state["m_neg"] = 1.0

    if 'slip_pos' not in st.session_state:
        st.session_state["slip_pos"] = 0.0

    if 'slip_neg' not in st.session_state:
        st.session_state["slip_neg"] = 0.0

    if 's_neg' not in st.session_state:
        st.session_state['s_neg'] = 0.0
        
    if 's_pos' not in st.session_state:
        st.session_state['s_pos'] = 0.0
        
    

    # For dV/dQ, it is any cycle which has a rate of C/20 or longer
    if plot_opts == 'dV/dQ':
        
        fastest_checkup = st.sidebar.text_input("Fastest Checkup Cycle (Default is C/20)", value="C/20").upper()
        
        # rates is a list which holds all rates which are C/20 or longer
        rates = dVdQ_rates(uf_rates)
        
        if len(rates) == 0:
            st.error("This file has no c-rates of " + fastest_checkup + " or slower (which is required for dV/dQ Analysis)")
            st.stop()
            

        # Controls for adjusting plot axes and for toggling between scatter and line plots
        dvdq_plot_expander = st.sidebar.expander("dV/dQ Plot Control")
        dqdv_plot_expander = st.sidebar.expander("dQ/dV Plot Control")

        with dvdq_plot_expander:
            dvdq_plot_type = st.radio("Do you want dV/dQ vs. Q to be a line or scatter plot?", ('Line', 'Scatter'))

        with dqdv_plot_expander:
            plot_dqdv = st.checkbox("Plot dQ/dV vs. V?")
            plot_view = st.radio("Vertical or Horizontal View?", ("Vertical", "Horizontal"))
            dqdv_plot_type = st.radio("Do you want dQ/dV vs. V to be a line or scatter plot?", ("Line", "Scatter"))


        range_or_individual = st.sidebar.radio("Fit over range of cycles or individual cycle?", ["Individual", "Range"])

        # Dropdown with selectable cycle rates (based on the 'rates' list)
        rate = st.sidebar.selectbox("Which C-rate would you like to see?",
                                    tuple(rates))

        # Message will appear if no rates of C/20 or longer are available
        if len(rates) == 0:
            st.sidebar.write("dV/dQ analysis cannot be performed because this file does not contain"
                             " cycle rates of C/20 or longer.")

        # dVdQ will be available if there are C/20 (or longer) rates available
        else:

            # Array of cycle numbers at the selected rate
            cyc_nums = np.array(selectByRate(rate))

            # The slider only includes cycle numbers from the selected rate
            cycnum = st.sidebar.select_slider("Select cycle to analyze.", options=list(cyc_nums))
            num_cycs = 1
            cap_m, volt_m = nd.get_vcurve(cycnum=cycnum)
            Q_meas, dVdQ_meas = dVdQ_m(cap_m, volt_m)

            V_meas, dQdV_meas = dQdV_m(cap_m, volt_m)


            st.write("Plotting cycle {0} with rate {1}:".format(cycnum, rate))

            # Setting up session state

            if 'fit_cap_min_i' not in st.session_state:
                st.session_state["fit_cap_min_i"] = int(min(Q_meas))

            if 'fit_cap_max_i' not in st.session_state:
                st.session_state["fit_cap_max_i"] = int(max(Q_meas))

            if 'fit_cap_min_f' not in st.session_state:
                st.session_state["fit_cap_min_f"] = int(min(Q_meas))

            if 'fit_cap_max_f' not in st.session_state:
                st.session_state["fit_cap_max_f"] = int(max(Q_meas))

            if 'fit_min' not in st.session_state:
                st.session_state["fit_min"] = int(min(Q_meas))

            if 'fit_max' not in st.session_state:
                st.session_state["fit_max"] = int(max(Q_meas))

            if 'window_size' not in st.session_state:
                st.session_state["window_size"] = 15

            if 'polyorder' not in st.session_state:
                st.session_state["polyorder"] = 4

            if 'dirname' not in st.session_state:
                st.session_state["dirname"] = ""

            if 'slip_neg_min' not in st.session_state:
                st.session_state["slip_neg_min"] = -45

            if 'slip_neg_max' not in st.session_state:
                st.session_state["slip_neg_max"] = 5

            if 'slip_neg_spacing' not in st.session_state:
                st.session_state["slip_neg_spacing"] = 2.0

            if 'slip_pos_min' not in st.session_state:
                st.session_state["slip_pos_min"] = -45

            if 'slip_pos_max' not in st.session_state:
                st.session_state["slip_pos_max"] = 5

            if 'slip_pos_spacing' not in st.session_state:
                st.session_state["slip_pos_spacing"] = 2.0

            if 'mass_neg_min' not in st.session_state:
                st.session_state["mass_neg_min"] = 0.9

            if 'mass_neg_max' not in st.session_state:
                st.session_state["mass_neg_max"] = 1.4

            if 'mass_neg_spacing' not in st.session_state:
                st.session_state["mass_neg_spacing"] = 0.1

            if 'mass_pos_min' not in st.session_state:
                st.session_state["mass_pos_min"] = 0.9

            if 'mass_pos_max' not in st.session_state:
                st.session_state["mass_pos_max"] = 1.4

            if 'mass_pos_spacing' not in st.session_state:
                st.session_state["mass_pos_spacing"] = 0.1
                

                

            # ========================================================================== #
            # Windows only feature #


            folder_expander = st.sidebar.expander("Select Folder for Saved Files")

            with folder_expander:

                # Folder picker button
                st.write('Please select a folder where your files will be saved to:')
                folder_button = st.button('Folder Picker')
                dirname = None
                if folder_button:
                    dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
                    st.session_state["dirname"] = dirname

            # ========================================================================== #


            # Expander for controlling the smoothing of the measured dVdQ curve
            smoothing_expander = st.sidebar.expander("Smoothing measured data")

            with smoothing_expander:
                smooth_cbox = st.checkbox('Smooth measured data', value=True)
                st.session_state["polyorder"] = st.number_input(label="Smoothing polynomial order "
                                                                      "(must be less than window size)", value=
                                                                st.session_state["polyorder"], min_value=1,
                                                                max_value=st.session_state["window_size"] - 1)

                st.session_state["window_size"] = st.slider(label="Window size",
                                                            min_value=st.session_state["polyorder"] + 1, max_value=31,
                                                            value=st.session_state["window_size"], step=2)

            # Only smooths measured data if checkbox is selected
            if smooth_cbox:
                dVdQ_meas = smooth_meas(dVdQ_meas, st.session_state["window_size"], st.session_state["polyorder"])
                dQdV_meas = smooth_meas(dQdV_meas, st.session_state["window_size"], st.session_state["polyorder"])

            locking_expander = st.expander("Locking fit parameters")

            with locking_expander:
                lock_pm = st.checkbox("Lock positive mass")

                lock_nm = st.checkbox("Lock negative mass")

                lock_ps = st.checkbox("Lock positive slippage")

                lock_ns = st.checkbox("Lock negative slippage")

            # Interpolating the reference data for calculating the dV/dQ curve

            dVdq_n, dVdq_p, q_range_p, q_range_n = interpolate_reference(v_n, q_n, v_p, q_p)

            brute_expander = st.sidebar.expander("Adjust brute force fit parameters")

            # Adjustment sliders for the brute force fit's parameters (in an expander)
            with brute_expander:

                st.markdown("""### Negative Slippage Controls""")
                ns_c1, ns_c2 = st.beta_columns(2)
                with ns_c1:
                    slip_neg_min = st.text_input(label="Neg. Slippage Minumum (mAh)", value=st.session_state["slip_neg_min"])
                    st.session_state["slip_neg_min"] = float(slip_neg_min)
                with ns_c2:
                    slip_neg_max = st.text_input(label="Neg. Slippage Maximum (mAh)", value=st.session_state["slip_neg_max"])
                    st.session_state["slip_neg_max"] = float(slip_neg_max)

                st.session_state["slip_neg_spacing"] = st.number_input(label="Negative slippage grid spacing (mAh)",
                                                                   value=st.session_state["slip_neg_spacing"])

                st.markdown("""### Positive Slippage Controls""")
                ps_c1, ps_c2 = st.beta_columns(2)
                with ps_c1:
                    slip_pos_min = st.text_input(label="Pos. Slippage Minumum (mAh)", value=st.session_state["slip_pos_min"])
                    st.session_state["slip_pos_min"] = float(slip_neg_min)
                with ps_c2:
                    slip_pos_max = st.text_input(label="Pos. Slippage Maximum (mAh)", value=st.session_state["slip_pos_max"])
                    st.session_state["slip_pos_max"] = float(slip_pos_max)

                st.session_state["slip_pos_spacing"] = st.number_input(label="Positive slippage grid spacing (mAh)",
                                                                   value=st.session_state["slip_pos_spacing"])

                st.markdown("""### Negative Mass Controls""")
                nm_c1, nm_c2 = st.beta_columns(2)
                with nm_c1:
                    mass_neg_min = st.text_input(label="Neg. Mass Minumum (g)", value=st.session_state["mass_neg_min"])
                    st.session_state["mass_neg_min"] = float(mass_neg_min)

                with nm_c2:
                    mass_neg_max = st.text_input(label="Neg. Mass Maximum (g)", value=st.session_state["mass_neg_max"])
                    st.session_state["mass_neg_max"] = float(mass_neg_max)

                st.session_state["mass_neg_spacing"] = st.number_input(label="Negative active mass grid spacing (g)",
                                                                   value=st.session_state["mass_neg_spacing"])

                st.markdown("""### Positive Mass Controls""")
                pm_c1, pm_c2 = st.beta_columns(2)
                with pm_c1:
                    mass_pos_min = st.text_input(label="Pos. Mass Minumum (g)", value=st.session_state["mass_pos_min"])
                    st.session_state["mass_pos_min"] = float(mass_pos_min)
                with pm_c2:
                    mass_pos_max = st.text_input(label="Pos. Mass Maximum (g)", value=st.session_state["mass_pos_max"])
                    st.session_state["pos_mass_max"] = float(mass_pos_max)

                st.session_state["mass_pos_spacing"] = st.number_input(label="Positive Active Mass Grid Spacing (g)",
                                                                   value=st.session_state["mass_pos_spacing"])

            if range_or_individual == "Individual":
                # Expander for specifying the capacity range over which the fit will work
                fit_range_expander = st.sidebar.expander("Fit over specified range")

                with fit_range_expander:
                    fit_range_cbox = st.checkbox('Fit over specified range')
                    cap_c1, cap_c2 = st.beta_columns(2)
                    with cap_c1:
                        fit_min = st.text_input(label="Minimum fit capacity (mAh)", value=st.session_state["fit_min"])
                        st.session_state["fit_min"] = float(fit_min)
                    with cap_c2:
                        fit_max = st.text_input(label="Maximum fit capacity (mAh)", value=st.session_state["fit_max"])
                        st.session_state["fit_max"] = float(fit_max)
                    

                # Indices over which the fit will be calculated
                fit_inds = np.where((Q_meas >= st.session_state["fit_min"]) & (Q_meas <= st.session_state["fit_max"]))[0]

                if fit_range_cbox:
                    Q_meas_fit = Q_meas[fit_inds]
                    dVdQ_meas_fit = dVdQ_meas[fit_inds]

                else:
                    Q_meas_fit = Q_meas
                    dVdQ_meas_fit = dVdQ_meas



                # Columns are for visually organizing the buttons on the screen
                col1, col2, col3 = st.beta_columns(3)

                with col1:
                    fit_button = st.button('Least Squares Fit')
                with col2:
                    brute_fit_button = st.button("Brute Force Fit")
                with col3:
                    brute_and_ls_button = st.button("Brute Force + Least Squares")


                if fit_button:

                    # Sometimes least squares can't find any better parameters and throws an error, hence why a "try" is used
                    ls_spinner = st.spinner("Least squares in progress")
                    with ls_spinner:
                        try:
                            [st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"],
                             st.session_state["m_neg"]] = \
                                least_squares_fit(Q_meas_fit, dVdQ_meas_fit, st.session_state["slip_pos"],
                                                  st.session_state["slip_neg"], st.session_state["m_pos"],
                                                  st.session_state["m_neg"],
                                                  -100., -100., 0.5, 0.5, 50., 50., 2., 2.)

                            dVdQ_calc = dVdQ_fitting(Q_meas, [st.session_state["slip_pos"], st.session_state["slip_neg"],
                                                              st.session_state["m_pos"], st.session_state["m_neg"]])
                            Q = Q_meas

                            V, dQdV_calc = dQdV_c(st.session_state["slip_pos"], st.session_state["slip_neg"],
                                                  st.session_state["m_pos"],
                                                  st.session_state["m_neg"])

                        except:
                            fit_button = False

                elif brute_fit_button:
                    dVdQ_calc, st.session_state["m_neg"], st.session_state["m_pos"], st.session_state["slip_neg"], \
                    st.session_state["slip_pos"] = brute_force_fit(st.session_state["m_pos"], st.session_state["mass_pos_min"],
                                                               st.session_state["mass_pos_max"],
                                                               st.session_state["mass_pos_spacing"],
                                                               st.session_state["m_neg"], st.session_state["mass_neg_min"],
                                                               st.session_state["mass_neg_max"],
                                                               st.session_state["mass_neg_spacing"],
                                                               st.session_state["slip_pos"], st.session_state["slip_pos_min"],
                                                               st.session_state["slip_pos_max"],
                                                               st.session_state["slip_pos_spacing"],
                                                               st.session_state["slip_neg"], st.session_state["slip_neg_min"],
                                                               st.session_state["slip_neg_max"],
                                                               st.session_state["slip_neg_spacing"],
                                                               dVdQ_meas_fit, Q_meas_fit)

                    dVdQ_calc = dVdQ_fitting(Q_meas, st.session_state["slip_pos"], st.session_state["slip_neg"],
                                             st.session_state["m_pos"], st.session_state["m_neg"])

                    Q = Q_meas

                    V, dQdV_calc = dQdV_c(st.session_state["slip_pos"], st.session_state["slip_neg"],
                                          st.session_state["m_pos"],
                                          st.session_state["m_neg"])

                elif brute_and_ls_button:
                    dVdQ_calc, st.session_state["m_neg"], st.session_state["m_pos"], st.session_state["slip_neg"], \
                    st.session_state["slip_pos"] = brute_force_fit(st.session_state["m_pos"], st.session_state["mass_pos_min"],
                                                               st.session_state["mass_pos_max"],
                                                               st.session_state["mass_pos_spacing"],
                                                               st.session_state["m_neg"], st.session_state["mass_neg_min"],
                                                               st.session_state["mass_neg_max"],
                                                               st.session_state["mass_neg_spacing"],
                                                               st.session_state["slip_pos"], st.session_state["slip_pos_min"],
                                                               st.session_state["slip_pos_max"],
                                                               st.session_state["slip_pos_spacing"],
                                                               st.session_state["slip_neg"], st.session_state["slip_neg_min"],
                                                               st.session_state["slip_neg_max"],
                                                               st.session_state["slip_neg_spacing"],
                                                               dVdQ_meas_fit, Q_meas_fit)



                    # Sometimes least squares can't find any better parameters and throws an error, hence why a "try" is used
                    try:
                        [st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"],
                         st.session_state["m_neg"]] = \
                            least_squares_fit(Q_meas_fit, dVdQ_meas_fit, st.session_state["slip_pos"],
                                              st.session_state["slip_neg"], st.session_state["m_pos"],
                                              st.session_state["m_neg"],
                                              -100., -100., 0.5, 0.5, 50., 50., 2., 2.)

                        dVdQ_calc = dVdQ_fitting(Q_meas,
                                                 [st.session_state["slip_pos"], st.session_state["slip_neg"],
                                                  st.session_state["m_pos"],
                                                  st.session_state["m_neg"]])
                        Q = Q_meas
                    except:
                        Q, dVdQ_calc = dVdQ_c(st.session_state["slip_pos"], st.session_state["slip_neg"],
                                              st.session_state["m_pos"],
                                              st.session_state["m_neg"])
                        V, dQdV_calc = dQdV_c(st.session_state["slip_pos"], st.session_state["slip_neg"],
                                              st.session_state["m_pos"],
                                              st.session_state["m_neg"])

                slider_expander = st.sidebar.expander("Adjust active mass and slippages")

                with slider_expander:
                    st.session_state["m_pos"] = st.number_input("Positive Mass (g)", value=st.session_state["m_pos"])
                    st.session_state["m_neg"] = st.number_input("Negative Mass (g)", value=st.session_state["m_neg"])
                    st.session_state["slip_pos"] = st.number_input("Positive Slippage (mAh)", value=st.session_state["slip_pos"])
                    st.session_state["slip_neg"] = st.number_input("Negative Slippage (mAh)", value=st.session_state["slip_neg"])

                # An "if not" had to be used instead of an else because the sliders would only behave properly if they
                #   followed the first if statement and preceded the next!

                if not fit_button and not brute_fit_button:
                    Q, dVdQ_calc = dVdQ_c(st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"],
                                          st.session_state["m_neg"])
                    V, dQdV_calc = dQdV_c(st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"],
                                          st.session_state["m_neg"])

            if range_or_individual == "Range":

                fit_range_cbox = False

                intermediate_fits = st.sidebar.expander("Plot intermediate fits")

                multi_fit_expander = st.sidebar.expander("Fit Over Specified Range")

                plot_parameter_expander = st.sidebar.expander("Plot parameters vs. cycle number")

                with plot_parameter_expander:
                    display_parameter_plots = st.checkbox("Display fit parameters vs. cycle number")

                with intermediate_fits:
                    #============== Windows only ================#
                    export_int_plots = st.checkbox("Export Plots Over Intervals?")
                    export_plot_bool = export_int_plots
                    #============================================#
                    # Mac only
                    #export_plot_bool = False
                    # ============================================#

                    freq_int_plots = st.number_input("Interval of intermediate fit plots", value=5, min_value=1)
                    display_int_plots = st.checkbox("Display intermediate fits")
                    display_plots_bool = display_int_plots

                with multi_fit_expander:
                    adjust_range_fit = st.checkbox("Adjust Range Fit Bounds?")

                    fit_num_range = st.select_slider("Cycle numbers to be fit:", options=list(cyc_nums), value=(int(min(cyc_nums)),
                                                                                                         int(max(
                                                                                                             cyc_nums))))
                    if adjust_range_fit:

                        cap_range_radio = st.radio("Which range would you like to adjust for the autofit?", ["First Cycle", "Last Cycle"])

                        if cap_range_radio == "First Cycle":
                            cap_m, volt_m = nd.get_vcurve(cycnum=fit_num_range[0])
                            Q_meas, dVdQ_meas = dVdQ_m(cap_m, volt_m)
                            V_meas, dQdV_meas = dQdV_m(cap_m, volt_m)

                            fit_col_i_1, fit_col_i_2 = st.beta_columns(2)

                            with fit_col_i_1:
                                st.session_state["fit_cap_min_i"] = int(st.text_input(label="Minimum capacity (First Cycle)", value=st.session_state["fit_cap_min_i"]))

                            with fit_col_i_2:
                                st.session_state["fit_cap_max_i"] = int(st.text_input(label="Maximum capacity (First Cycle)", value=st.session_state["fit_cap_max_i"]))

                            st.session_state["fit_min"] = st.session_state["fit_cap_min_i"]
                            st.session_state["fit_max"] = st.session_state["fit_cap_max_i"]

                        elif cap_range_radio == "Last Cycle":
                            cap_m, volt_m = nd.get_vcurve(cycnum=fit_num_range[1])
                            Q_meas, dVdQ_meas = dVdQ_m(cap_m, volt_m)
                            V_meas, dQdV_meas = dQdV_m(cap_m, volt_m)

                            fit_col_f_1, fit_col_f_2 = st.beta_columns(2)

                            with fit_col_f_1:
                                st.session_state["fit_cap_min_f"] = int(
                                    st.text_input(label="Minimum capacity (Last Cycle)",
                                                  value=st.session_state["fit_cap_min_f"]))

                            with fit_col_f_2:
                                st.session_state["fit_cap_max_f"] = int(
                                    st.text_input(label="Maximum capacity (Last Cycle)",
                                                  value=st.session_state["fit_cap_max_f"]))

                            st.session_state["fit_min"] = st.session_state["fit_cap_min_f"]
                            st.session_state["fit_max"] = st.session_state["fit_cap_max_f"]

                    file_name = st.text_input("Save parameters to .txt file with name (do not add the '.txt' suffix):")

                    file_tag = st.text_input("File tag to be used as the header to the file:")

                    multi_fit_button = st.button("Start range fit")

                neg_slip_arr = []
                pos_slip_arr = []
                m_pos_arr = []
                m_neg_arr = []

                cap_m_i, volt_m_i = nd.get_vcurve(cycnum=fit_num_range[0])
                Q_meas_i, dVdQ_meas_i = dVdQ_m(cap_m_i, volt_m_i)

                # Final fit cycle in the specified range
                cap_m_f, volt_m_f = nd.get_vcurve(cycnum=fit_num_range[-1])
                Q_meas_f, dVdQ_meas_f = dVdQ_m(cap_m_f, volt_m_f)

                range_inds = np.where((list(cyc_nums) >= fit_num_range[0]) & (list(cyc_nums) <= fit_num_range[-1]))

                fit_inds = np.where((Q_meas_i >= st.session_state["fit_cap_min_i"]) & (Q_meas_i <= st.session_state["fit_cap_max_f"]))[0]

                if len(fit_inds) == 0:
                    cap_range_first_ind = 0
                    cap_range_last_ind = len(Q_meas_i) - 1

                else:
                    cap_range_first_ind = fit_inds[0]
                    cap_range_last_ind = fit_inds[-1]



                Q_meas_f_t = Q_meas_f[cap_range_first_ind: cap_range_last_ind]
                dVdQ_meas_f_t = dVdQ_meas_f[cap_range_first_ind: cap_range_last_ind]

                Q_meas_i_t = Q_meas_i[cap_range_first_ind: cap_range_last_ind]
                dVdQ_meas_i_t = dVdQ_meas_i[cap_range_first_ind: cap_range_last_ind]


                if smooth_cbox:

                    dVdQ_meas_f_t = smooth_meas(dVdQ_meas_f_t, st.session_state["window_size"],
                                                     st.session_state["polyorder"])
                    dVdQ_meas_i_t = smooth_meas(dVdQ_meas_i_t, st.session_state["window_size"],
                                                     st.session_state["polyorder"])

                    dVdQ_meas_f = smooth_meas(dVdQ_meas_f, st.session_state["window_size"],
                                                   st.session_state["polyorder"])
                    dVdQ_meas_i = smooth_meas(dVdQ_meas_i, st.session_state["window_size"],
                                                   st.session_state["polyorder"])
                    dVdQ_meas = smooth_meas(dVdQ_meas, st.session_state["window_size"],
                                                   st.session_state["polyorder"])

                if multi_fit_button:

                    #=========================================================#
                    # Windows only version#

                    if st.session_state["dirname"] == "":
                        file = open(file_name + ".txt", "w")
                    else:
                        file = open(st.session_state["dirname"][2:] + "/" + str(file_name) + ".txt", "w")
                    # =========================================================#

                    # ==========================================#
                    # Mac Version
                    #file = open(str(file_name) + ".txt", "w")
                    # ==========================================#

                    file.write(file_tag + "\n")
                    file.write("Cycle Number  Negative Slippage (mAh)  Positive Slippage (mAh)  Negative Mass (g) Positive Mass (g) Shift Loss (mAh)" + "\n")

                    dVdQ_calc, st.session_state["m_neg"], st.session_state["m_pos"], st.session_state["slip_neg"], \
                    st.session_state["slip_pos"] = brute_force_fit(st.session_state["m_pos"],
                                                               st.session_state["mass_pos_min"],
                                                               st.session_state["mass_pos_max"],
                                                               st.session_state["mass_pos_spacing"],
                                                               st.session_state["m_neg"],
                                                               st.session_state["mass_neg_min"],
                                                               st.session_state["mass_neg_max"],
                                                               st.session_state["mass_neg_spacing"],
                                                               st.session_state["slip_pos"],
                                                               st.session_state["slip_pos_min"],
                                                               st.session_state["slip_pos_max"],
                                                               st.session_state["slip_pos_spacing"],
                                                               st.session_state["slip_neg"],
                                                               st.session_state["slip_neg_min"],
                                                               st.session_state["slip_neg_max"],
                                                               st.session_state["slip_neg_spacing"],
                                                               dVdQ_meas_f_t, Q_meas_f_t)

                    try:
                        st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"], st.session_state["m_neg"] = \
                            least_squares_fit(Q_meas_f_t, dVdQ_meas_f_t, st.session_state["slip_pos"], st.session_state["slip_neg"],
                                                  st.session_state["m_pos"],
                                                  st.session_state["m_neg"], -100., -100., 0.5, 0.5, 50., 50., 2., 2.)

                    except:
                        pass

                    sp_f, sn_f, mp_f, mn_f = st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"], \
                                             st.session_state["m_neg"]

                    dVdQ_calc, st.session_state["m_neg"], st.session_state["m_pos"], st.session_state["slip_neg"], \
                        st.session_state["slip_pos"] = brute_force_fit(st.session_state["m_pos"],
                                                                   st.session_state["mass_pos_min"],
                                                                   st.session_state["mass_pos_max"],
                                                                   st.session_state["mass_pos_spacing"],
                                                                   st.session_state["m_neg"],
                                                                   st.session_state["mass_neg_min"],
                                                                   st.session_state["mass_neg_max"],
                                                                   st.session_state["mass_neg_spacing"],
                                                                   st.session_state["slip_pos"],
                                                                   st.session_state["slip_pos_min"],
                                                                   st.session_state["slip_pos_max"],
                                                                   st.session_state["slip_pos_spacing"],
                                                                   st.session_state["slip_neg"],
                                                                   st.session_state["slip_neg_min"],
                                                                   st.session_state["slip_neg_max"],
                                                                   st.session_state["slip_neg_spacing"],
                                                                   dVdQ_meas_i_t, Q_meas_i_t)


                    try:
                        st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"], st.session_state[
                            "m_neg"] = st.session_state["slip_pos"], st.session_state["slip_neg"], \
                                       st.session_state["m_pos"], st.session_state["m_neg"] = \
                            least_squares_fit(Q_meas_i_t, dVdQ_meas_i_t, st.session_state["slip_pos"],
                                              st.session_state["slip_neg"],
                                          st.session_state["m_pos"],
                                          st.session_state["m_neg"],
                                          -100., -100., 0.5, 0.5, 50., 50., 2., 2.)

                    except:
                        pass

                    mp_i, mn_i, sp_i, sn_i = st.session_state["m_pos"], st.session_state["m_neg"], st.session_state["slip_pos"], \
                                             st.session_state["slip_neg"]



                    dVdQ_calc_i = plotting(Q_meas_i, dVdQ_meas_i, fit_num_range[0], export_plot_bool, display_plots_bool)

                    min_mp = min(mp_i, mp_f) - (0.1 * abs(mp_f - mp_i))
                    max_mp = max(mp_i, mp_f) + (0.1 * abs(mp_f - mp_i))
                    if min_mp == max_mp:

                        min_mp = min_mp - abs(0.1 * min_mp)
                        max_mp = max_mp + abs(0.1 * max_mp)

                    min_mn = min(mn_i, mn_f) - (0.1 * abs(mn_f - mn_i))
                    max_mn = max(mn_i, mn_f) + (0.1 * abs(mn_f - mn_i))

                    if min_mn == max_mn:
                        min_mn = min_mn - abs(0.1 * min_mn)
                        max_mn = max_mn + abs(0.1 * max_mn)

                    min_sp = min(sp_i, sp_f) - (0.1 * abs(sp_f - sp_i))
                    max_sp = max(sp_i, sp_f) + (0.1 * abs(sp_f - sp_i))
                    if min_sp == max_sp:
                        min_sp = min_sp - abs(0.1 * min_sp)
                        max_sp = max_sp + abs(0.1 * max_sp)

                    min_sn = min(sn_i, sn_f) - (0.1 * abs(sn_f - sn_i))
                    max_sn = max(sn_i, sn_f) + (0.1 * abs(sn_f - sn_i))
                    if min_sn == max_sn:
                        min_sn = min_sn - abs(0.1 * min_sn)
                        max_sn = max_sn + abs(0.1 * max_sn)

                    file.write(str(round(fit_num_range[0],2)) + "  " + str(round(sn_i, 2)) + "  " + str(round(sp_i, 2)) +
                               "  " + str(round(mn_i, 2)) + "  " + str(round(mp_i,2)) + "\n")

                    pos_slip_arr.append(round(sp_i, 2))
                    neg_slip_arr.append(round(sn_i, 2))
                    m_pos_arr.append(round(mp_i, 2))
                    m_neg_arr.append(round(mn_i, 2))


                    min_bounds = np.linspace(st.session_state["fit_cap_min_i"], st.session_state["fit_cap_min_f"],
                                             len(range_inds[0]))[1:]
                    max_bounds = np.linspace(st.session_state["fit_cap_max_i"], st.session_state["fit_cap_max_f"],
                                             len(range_inds[0]))[1:]

                    range_count = 0
                    for cn in list(cyc_nums[range_inds[0][1:]]):
                        cap_m, volt_m = nd.get_vcurve(cycnum=cn)
                        Q_me, dVdQ_me = dVdQ_m(cap_m, volt_m)

                        fit_inds = np.where(
                            (Q_me >= min_bounds[range_count]) & (Q_me <= max_bounds[range_count]))[
                            0]

                        cap_range_first_ind = fit_inds[0]
                        cap_range_last_ind = fit_inds[-1]

                        Q_me_t = Q_me[cap_range_first_ind: cap_range_last_ind]
                        dVdQ_me_t = dVdQ_me[cap_range_first_ind: cap_range_last_ind]


                        if smooth_cbox:
                            dVdQ_me_t = smooth_meas(dVdQ_me_t, st.session_state["window_size"],
                                                         st.session_state["polyorder"])
                            dVdQ_me = smooth_meas(dVdQ_me, st.session_state["window_size"],
                                                         st.session_state["polyorder"])

                        if range_count == int(len(range_inds[0])/2):
                            dVdQ_calc, st.session_state["m_neg"], st.session_state["m_pos"], st.session_state["slip_neg"], \
                            st.session_state["slip_pos"] = brute_force_fit(st.session_state["m_pos"],
                                                                       st.session_state["mass_pos_min"],
                                                                       st.session_state["mass_pos_max"],
                                                                       st.session_state["mass_pos_spacing"],
                                                                       st.session_state["m_neg"],
                                                                       st.session_state["mass_neg_min"],
                                                                       st.session_state["mass_neg_max"],
                                                                       st.session_state["mass_neg_spacing"],
                                                                       st.session_state["slip_pos"],
                                                                       st.session_state["slip_pos_min"],
                                                                       st.session_state["slip_pos_max"],
                                                                       st.session_state["slip_pos_spacing"],
                                                                       st.session_state["slip_neg"],
                                                                       st.session_state["slip_neg_min"],
                                                                       st.session_state["slip_neg_max"],
                                                                       st.session_state["slip_neg_spacing"],
                                                                       dVdQ_me_t, Q_me_t)

                        try:
                            st.session_state["slip_pos"], st.session_state["slip_neg"], st.session_state["m_pos"], st.session_state["m_neg"] = \
                                least_squares_fit(Q_me_t, dVdQ_me_t, st.session_state["slip_pos"], st.session_state["slip_neg"],
                                                  st.session_state["m_pos"],
                                                  st.session_state["m_neg"],
                                                  min_sp, min_sn, min_mp, min_mn, max_sp, max_sn, max_mp, max_mn)
                        except:
                            pass

                        if range_count % freq_int_plots == 0:

                            dVdQ_meas = plotting(Q_me, dVdQ_me, cn, export_plot_bool, display_plots_bool)

                        file.write(str(cn) + "  " + str(round(st.session_state["slip_neg"], 2)) + "  " + str(round(st.session_state["slip_pos"], 2)) +

                                   "  " + str(round(st.session_state["m_neg"], 2)) + "  " + str(round(st.session_state["m_pos"], 2)) + "  " + str((pos_slip_arr[0] - neg_slip_arr[0]) - (round(st.session_state["slip_pos"], 2) - round(st.session_state["slip_neg"], 2))) + "\n")

                        pos_slip_arr.append(round(st.session_state["slip_pos"], 2))
                        neg_slip_arr.append(round(st.session_state["slip_neg"], 2))
                        m_pos_arr.append(round(st.session_state["m_pos"], 2))
                        m_neg_arr.append(round(st.session_state["m_neg"], 2))

                        range_count += 1


                    file.close()

                    if display_parameter_plots:

                        sn_fig = figure(plot_width=450, plot_height=250,
                           y_axis_label='Negative Slippage (mAh)', x_axis_type=None)
                        sn_line = sn_fig.line(list(cyc_nums[range_inds[0]]), neg_slip_arr, color="red")

                        sp_fig = figure(plot_width=450, plot_height=250,
                                        y_axis_label='Positive Slippage (mAh)',
                                        x_range=sn_fig.x_range, x_axis_type=None)

                        sp_line = sp_fig.line(list(cyc_nums[range_inds[0]]), pos_slip_arr, color="red")

                        mn_fig = figure(plot_width=450, plot_height=250,
                                        y_axis_label='Negative Mass (g)',
                                        x_range=sn_fig.x_range, x_axis_type=None)

                        mn_line = mn_fig.line(list(cyc_nums[range_inds[0]]), m_neg_arr, color="red")

                        mp_fig = figure(plot_width=450, plot_height=250,
                                        y_axis_label='Positive Mass (g)',
                                        x_range=sn_fig.x_range)

                        mp_line = mp_fig.line(list(cyc_nums[range_inds[0]]), m_pos_arr, color="red")

                        gp = gridplot([[sn_fig], [sp_fig], [mn_fig], [mp_fig]])

                        st.bokeh_chart(gp)

                Q = Q_meas
                V = V_meas


        with dvdq_plot_expander:
            # Controls for the plot's axes
            st.write("X axis range:")
            # If the calculated dVdQ curve exists, then the default x axis minimum is either the lowest x value on the
            #   calculated or 0, whichever is lower
            if (len(Q_meas) > 0):
                dvdq_xmin = st.number_input('dV/dQ X-minimum', value=min(Q_meas))
            else:
                dvdq_xmin = 0
            dvdq_xlim = st.number_input('dV/dQ X-limit', value=max(Q_meas))
            st.write("Y axis range:")
            dvdq_ymin = st.number_input('dV/dQ Y-minimum')
            dvdq_ylim = st.number_input('dV/dQ Y-limit', value=0.015)

        with dqdv_plot_expander:
            # Controls for the plot's axes
            st.write("X axis range:")
            # If the calculated dVdQ curve exists, then the default x axis minimum is either the lowest x value on the
            #   calculated or 0, whichever is lower

            if (len(V) > 0):
                dqdv_xmin = st.number_input('dQ/dV X-minimum', value=min(min(V), min(V_meas)))
            else:
                dvdq_xmin = 0
            dqdv_xlim = st.number_input('dQ/dV X-limit', value=max(max(V), max(V_meas)))
            st.write("Y axis range:")
            dqdv_ymin = st.number_input('dQ/dV Y-minimum')
            if range_or_individual == "Individual":
                dqdv_ylim = st.number_input('dQ/dV Y-limit', value=max(max(dQdV_meas), max(dQdV_calc)) + 40)
            else:
                dqdv_ylim = st.number_input('dQ/dV Y-limit', value=max(dQdV_meas) + 40)



        dvdq_plot = figure(plot_width=600, x_range=(dvdq_xmin, dvdq_xlim), y_range=(dvdq_ymin, dvdq_ylim), plot_height=400,
                   x_axis_label='Capacity, Q (mAh)',
                   y_axis_label='dV/dQ (V/mAh)')
        dqdv_plot = figure(plot_width=600, plot_height=400, x_range=(dqdv_xmin, dqdv_xlim), y_range=(dqdv_ymin, dqdv_ylim),
                   x_axis_label='Voltage (V)',
                   y_axis_label='dQ/dV (mAh/V)')

        if ((range_or_individual == "Range") and adjust_range_fit) or fit_range_cbox:
            b = dvdq_plot.vbar(x=[(st.session_state["fit_max"] + st.session_state["fit_min"])/2], width=(st.session_state["fit_max"] -
                                                                                     st.session_state["fit_min"]),
                   bottom=0, top=10, color=['grey'], alpha=0.3)

        #dVdQ_calc = smooth_meas(dVdQ_calc, 35, st.session_state["polyorder"])

        if range_or_individual == "Individual":

            if dvdq_plot_type == 'Line':
                c_dvdq = dvdq_plot.line(Q, dVdQ_calc)
                m_dvdq = dvdq_plot.line(Q_meas, dVdQ_meas, color="red")

            elif dvdq_plot_type == 'Scatter':
                c_dvdq = dvdq_plot.circle(Q, dVdQ_calc)
                m_dvdq = dvdq_plot.circle(Q_meas, dVdQ_meas, color="red")

            if dqdv_plot_type == 'Line':

                c_dqdv = dqdv_plot.line(V, dQdV_calc)
                m_dqdv = dqdv_plot.line(V_meas, dQdV_meas, color="red")

            elif dqdv_plot_type == 'Scatter':

                c_dqdv = dqdv_plot.circle(V, dQdV_calc)
                m_dqdv = dqdv_plot.circle(V_meas, dQdV_meas, color="red")

            if fit_range_cbox:
                legend_dvdq = Legend(items=[
                    LegendItem(label="Calculated dV/dQ", renderers=[c_dvdq], index=0),
                    LegendItem(label="Measured dV/dQ", renderers=[m_dvdq], index=1),
                    LegendItem(label="Fit range", renderers=[b], index=2)
                ])

            else:
                legend_dvdq = Legend(items=[
                    LegendItem(label="Calculated dV/dQ", renderers=[c_dvdq], index=0),
                    LegendItem(label="Measured dV/dQ", renderers=[m_dvdq], index=1)
                ])

            legend_dqdv = Legend(items=[
                LegendItem(label="Calculated dQ/dV", renderers=[c_dqdv], index=0),
                LegendItem(label="Measured dQ/dV", renderers=[m_dqdv], index=1),
            ])

            dvdq_plot.add_layout(legend_dvdq)
            dqdv_plot.add_layout(legend_dqdv)


            if plot_view == "Horizontal" and plot_dqdv:

                p = gridplot([[dvdq_plot, dqdv_plot]])
                st.bokeh_chart(p, use_container_width=True)

            elif plot_view == "Vertical":
                dvdq_bokeh_chart = st.bokeh_chart(dvdq_plot, use_container_width=True)

                if plot_dqdv:
                    dqdv_bokeh_chart = st.bokeh_chart(dqdv_plot, use_container_width=True)
            else:
                dvdq_bokeh_chart = st.bokeh_chart(dvdq_plot, use_container_width=True)


        elif range_or_individual == "Range" and not multi_fit_button:
            if adjust_range_fit:
                if dvdq_plot_type == 'Line':
                    m_dvdq = dvdq_plot.line(Q_meas, dVdQ_meas, color="red")
                    m_dqdv = dqdv_plot.line(V_meas, dQdV_meas, color="red")
                elif dvdq_plot_type == 'Scatter':
                    m_dvdq = dvdq_plot.circle(Q_meas, dVdQ_meas, color="black")
                    m_dqdv = dqdv_plot.circle(V_meas, dQdV_meas, color="black")

                legend_dvdq = Legend(items=[
                    LegendItem(label="Measured", renderers=[m_dvdq], index=0),
                    LegendItem(label="Fit range", renderers=[b], index=1)
                ])


            else:
                if dvdq_plot_type == 'Line':
                    m_dvdq = dvdq_plot.line(Q_meas, dVdQ_meas, color="red")
                    m_dqdv = dqdv_plot.line(V_meas, dQdV_meas, color="red")
                elif dvdq_plot_type == 'Scatter':
                    m_dvdq = dvdq_plot.circle(Q_meas, dVdQ_meas, color="black")
                    m_dqdv = dqdv_plot.circle(V_meas, dQdV_meas, color="black")

                legend_dvdq = Legend(items=[
                    LegendItem(label="Measured dV/dQ", renderers=[m_dvdq], index=0)])

            legend_dqdv = Legend(items=[
                LegendItem(label="Measured dQ/dV", renderers=[m_dqdv], index=0),
            ])

            dvdq_plot.add_layout(legend_dvdq)
            dqdv_plot.add_layout(legend_dqdv)

            st.bokeh_chart(dvdq_plot, use_container_width=True)

            if plot_dqdv:
                st.bokeh_chart(dqdv_plot, use_container_width=True)

    if plot_opts == 'Cell Explorer':
        
        if posData is not None and negData is not None:
            cell_ex_sel = st.sidebar.radio("What would you like to see?", ["V-Q", "dQ/dV vs. V", "dV/dQ of Interpolated Reference Data"])
        else:
            cell_ex_sel = st.sidebar.radio("What would you like to see?", ["V-Q", "dQ/dV vs. V"])
        
        
        if cell_ex_sel != 'dV/dQ of Interpolated Reference Data':
                
            indiv_or_mult = st.sidebar.radio("Display one cycle or multiple at once?", ["One Cycle", "Multiple Cycles"])
                
            rates = ['All'] + nd.get_rates()
            rate = st.sidebar.selectbox("Which C-rate would you like to see?",
                                            tuple(rates))
        
                
                
            ncycs = nd.get_ncyc()
            if rate == 'All':
                cyc_nums = np.arange(1, ncycs + 1)
            else:
                cyc_nums = np.array(nd.select_by_rate(rate))
            
            if indiv_or_mult == "Multiple Cycles":
    
                # Slider that determines which cycles are displayed depends on which cycle rate was selected (adjusts to only
                #   include cycles that were done at the selected rate)
                cyc_range = st.sidebar.select_slider("Cycle Numbers", options=list(cyc_nums), value=(int(min(cyc_nums)),
                                                                                                     int(max(cyc_nums))))
                inds = np.where((cyc_nums <= cyc_range[1]) & (cyc_nums >= cyc_range[0]))[0]
                cycnums = cyc_nums[inds]
                num_cycs = len(cycnums)
    
                st.write("Plotting {0} cycles within range: ({1}, {2})".format(rate,
                                                                               cyc_range[0],
                                                                               cyc_range[1]))
            else:
                cycle = st.sidebar.select_slider("Cycle Numbers", options=list(cyc_nums), value=(int(min(cyc_nums))))
                num_cycs = 1
                cycnums = list([cycle])
                st.write("Plotting cycle: {0}".format(cycnums[0]))
                
            # Colour options not working
            #cmap = st.sidebar.selectbox("Color pallette",
            #                            ('Default', 'viridis', 'cividis'))
            #if cmap == 'Default':
            #    avail_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            #    colors = avail_colors * int(num_cycs / len(avail_colors) + 1)
            #elif cmap == 'viridis':
            #    colors = bp.viridis(num_cycs)
            #elif cmap == 'cividis':
            #    colors = bp.cividis(num_cycs)
    
            # For now, no colour options
            if num_cycs > 255:
                avail_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                colors = avail_colors * int(num_cycs / len(avail_colors) + 1)
            else:
                colors = bp.viridis(num_cycs)
            
            if cell_ex_sel == "dQ/dV vs. V":
    
                if 'window_size' not in st.session_state:
                    st.session_state["window_size"] = 15
        
                if 'polyorder' not in st.session_state:
                    st.session_state["polyorder"] = 4
        
                # Expander for controlling the smoothing of the measured dVdQ curve
                smoothing_expander = st.sidebar.expander("Smoothing measured data")
        
                with smoothing_expander:
                    smooth_cbox = st.checkbox('Smooth measured data', value=True)
                    
                    if smooth_cbox:
            
                        st.session_state["polyorder"] = st.number_input(label="Smoothing polynomial order "
                                                                              "(must be less than window size)", value=
                                                                        int(st.session_state["polyorder"]), min_value=1,
                                                                        max_value=int(st.session_state["window_size"] - 1.0), step=2)
            
                        st.session_state["window_size"] = st.slider(label="Window size", min_value=int(st.session_state["polyorder"] + 1), max_value=31,
                                                                    value=int(st.session_state["window_size"]), step=2)
            
        
        
                # When checkbox is selected, V-Q plot renders
                plot_cbox = st.sidebar.checkbox('Plot!')
        
                # If user selects the "plot" checkbox, plot will render given the predefined cycle numbers
                if plot_cbox:
    
        
                    p = figure(plot_width=600, plot_height=400,
                               x_axis_label='Potential (V)',
                               y_axis_label='dQ/dV (mAh/V)')
        
                    v_list, dqdv_list = dqdv_curves(cycnums, active_mass=None, explore=True)
        
                    for v, dqdv, color in zip(v_list, dqdv_list, colors):
                        if smooth_cbox:
                            dqdv = smooth_meas(dqdv, int(st.session_state["window_size"]), int(st.session_state["polyorder"]))
                        p.line(v, dqdv,color=color, line_width=2.0)
        
                    st.bokeh_chart(p)
        
            elif cell_ex_sel == 'V-Q':
    
                active_mass = st.sidebar.number_input("Active material mass (in grams):")
                if active_mass == 0.0:
                    active_mass = None
                else:
                    st.write("Calculating specific capacity using {} g active material".format(active_mass))
    
                # When checkbox is selected, V-Q plot renders
                plot_cbox = st.sidebar.checkbox('Plot!')
        
                # If user selects the "plot" checkbox, plot will render given the predefined cycle numbers
                if plot_cbox:
        
                    p = figure(plot_width=800, plot_height=400)
        
                    caps, volts = voltage_curves(cycnums, cyctype="cycle", active_mass=active_mass)
        
                    if active_mass is not None:
                        p.xaxis.axis_label = 'Specific Capacity (mAh/g)'
                    else:
                        p.xaxis.axis_label = 'Capacity (mAh)'
                    p.yaxis.axis_label = 'Voltage (V)'
                    for cap, volt, color in zip(caps, volts, colors):
                        p.line(cap, volt, color=color, line_width=2.0)
        
                    st.bokeh_chart(p, use_container_width=True)
        
                # ========== Windows only ===============#
                #rel_path = st.text_input("Save figure to: C://")
                #savepng_button = st.button("Save figure to png!")
                #savehtml_button = st.button("Save figure to html! (interactive plot)")
        
                #home_path = Path("/home/mmemc")
                #fig_path = home_path / rel_path
                #if savehtml_button is True:
                #    save(p, filename="{}.html".format(fig_path))
                #if savepng_button is True:
                #    export_png(p, filename="{}.png".format(fig_path))
                # =======================================#
                
        elif cell_ex_sel == 'dV/dQ of Interpolated Reference Data':
            ref_type = st.sidebar.radio("Positive or negative reference data?", ['Positive Reference', 'Negative Reference'])
            
            if ref_type == 'Positive Reference':
                #s_neg = st.session_state['s_neg']
                st.session_state['s_pos'] =  st.sidebar.number_input("s value for the interpolation of the positive reference data:",
                            value = st.session_state['s_pos'], step=1e-4,format="%.5f")
                st.sidebar.write('If noisy, increase s. Beware, s should be as small as possible. 1e-5 is reasonable.')
                
                
            else:
                #s_pos = st.session_state['s_pos']
                st.session_state['s_neg'] =  st.sidebar.number_input("s value for the interpolation of the negative reference data:",
                            value = st.session_state['s_neg'], step=1e-4,format="%.5f")
                st.sidebar.write('If noisy, increase s. Beware, s should be as small as possible. 1e-5 is reasonable.')
                
            
                
            with st.sidebar.expander("Plot Adjustments"):
                plot_type = st.radio("Line or scatter plot?", ("Line", "Scatter"))
                
                ref_ymin = st.number_input('dV/dQ Y-minimum', value = -0.01)
                ref_ylim = st.number_input('dV/dQ Y-limit', value=0.01)
                
                    
            with st.sidebar.expander("Active masses and slippages"):
                    
                st.session_state["m_pos"] = st.number_input("Positive Mass (g)", value=st.session_state["m_pos"])
                st.session_state["m_neg"] = st.number_input("Negative Mass (g)", value=st.session_state["m_neg"])
                st.session_state["slip_pos"] = st.number_input("Positive Slippage (mAh)", value=st.session_state["slip_pos"])
                st.session_state["slip_neg"] = st.number_input("Negative Slippage (mAh)", value=st.session_state["slip_neg"])

            
            ref_Q, ref_dVdQ = reference_dVdQ_c(ref_type, st.session_state['s_neg'], st.session_state['s_pos'])
            #st.session_state['s_neg'] = s_neg
           # st.session_state['s_pos'] = s_pos
            
            
            ref_fig = figure(plot_width=450, plot_height=250,
                           y_axis_label='dV/dQ (V/mAh)', x_axis_label='Q (mAh)', y_range=(ref_ymin, ref_ylim))
            
            if plot_type == 'Line':
                ref_dVdQ_curve = ref_fig.line(ref_Q, ref_dVdQ)
            if plot_type == 'Scatter':
                ref_dVdQ_curve = ref_fig.circle(ref_Q, ref_dVdQ)
            st.bokeh_chart(ref_fig, use_container_width=True)
            
            
            
            
            
            
            
            
            

            
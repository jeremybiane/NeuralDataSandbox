#!/usr/bin/env python
# conding : utf-8

import re
import os
import pickle
import xml.etree.ElementTree as ET
import json

import numpy as np
from scipy import stats as sstats
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.covariance import EmpiricalCovariance
from sklearn.svm import SVC
from statsmodels.sandbox.stats.multicomp import multipletests

def load_variable(name, folder='./'):
    with open(os.path.join(folder, name)) as f:
        toret = pickle.load(f)
    return toret

def load_cellreg_dict(name, session):
    with open(name, 'r') as f:
        toret = json.load(f)
    return np.r_[toret[session]][np.nonzero(np.prod(np.c_[[v for v in toret.itervalues()]], 0))]-1

def read_behavior(filename, sync_to_begin=False, begin_string='BEGIN'):
    try:
        with open(filename) as behavior_file:
            behavior = behavior_file.readlines()
    #         behavior = [[int(bb) for bb in b[:-2].split()] for b in behavior if b[0] != '#' and b[0] != '\r']
            behavior = [b.split() for b in behavior if b[0] != "#" and b[0] != "\r"]
            behavior = [[float(b[0])*1.e-3, b[1]] for b in behavior]
    except IOError, msg:
        print "Wait, I wasn't able to open the behavior.txt file, did you create one?"
        raise
    if sync_to_begin:
        start_2p = parse_behavior(behavior, begin_string)[0]
        behavior = [[float(b[0])-float(start_2p), b[1]] for b in behavior]  
    return behavior

def parse_behavior(behavior, text, offset=0):
    p = re.compile(text)
    return np.r_[[b[0]+offset for b in behavior if p.search(b[1])]]

def filter_cycle(time_ax, cycles, cycle):
    return (time_ax>=cycles[cycle][0]) * (time_ax<cycles[cycle][1])

def compute_all_dffs(time_ax, dff, cell, cycles, time_ax_single, cycle_filter=lambda x: True):
    return np.r_[[dff[:, cell][filter_cycle(time_ax, cycles, cycle)][:len(time_ax_single)]
                              for cycle in range(len(cycles)) if cycle_filter(cycle)]]

def compute_mean_traces(time_ax, dff, cycles, time_ax_single, cycle_filter=lambda x: True):
    traces_means = np.zeros((len(time_ax_single), dff.shape[1]))
    traces_std = np.zeros((len(time_ax_single), dff.shape[1]))
    for cell in xrange(dff.shape[1]):
        all_dffs = compute_all_dffs(time_ax, dff, cell, cycles, time_ax_single, cycle_filter)
        traces_means[:, cell] = all_dffs.mean(0)
        traces_std[:, cell] = np.std(all_dffs, 0)
    return traces_means, traces_std

def compute_baseline(time_ax, dff, base_start, base_stop):
    base_bool = (time_ax>=base_start) * (time_ax<base_stop)
    return dff[base_bool].mean(0)

def compute_auc_tone(time_ax, dff, cycles, time_ax_single, auc_baseline,
    tone_start=0, tone_duration=30):
    traces_means, traces_std = compute_mean_traces(time_ax, dff, cycles, time_ax_single)
    fps = 1./np.diff(time_ax)[0]
    fc = (time_ax_single>=tone_start) * (time_ax_single<tone_duration)    
    return np.sum((traces_means[fc]-auc_baseline), 0)/(4*fps)

def compute_auc_period(time_ax, dff, cycles, cycle, time_ax_single, cell, cycle_start=-10, start=0, end=30):
    t, dff_single = extract_single_cycle(time_ax, dff, cycles, cycle, cell, cycle_start=cycle_start)
    fps = 1./np.diff(time_ax)[0]
    fc = (time_ax_single>=start) * (time_ax_single<end)
    return np.sum(np.r_[dff_single][np.where(fc)])/fps

def compute_auc_pretone(time_ax, dff, cycles, time_ax_single, auc_baseline,
    tone_start=0, tone_duration=30, pretone_duration=30):
    traces_means, traces_std = compute_mean_traces(time_ax, dff, cycles, time_ax_single)
    fps = 1./np.diff(time_ax)[0]
    fc = (time_ax_single>=(tone_start-pretone_duration)) * (time_ax_single<tone_start)
    return np.sum((traces_means[fc]-auc_baseline), 0)/(4*fps)

def compute_lick_ratios(licks, cycles, cycle_start=-10, cs_start=0, cs_end=4, delay=4,
                        cs_duration=4, zero_value=-1):
    lick_ratios = []
    for s, e in cycles:
        l = licks - s + cycle_start
        licks_during = ((l>cs_start)*(l<cs_end+delay)).sum()
        licks_all = ((l>-cs_duration-delay)*(l<cs_end+delay)).sum()
        lick_ratios.append(1.*licks_during/licks_all if licks_all>0 else zero_value)
    return np.r_[lick_ratios]

def compute_licks_during(licks, cycles, start=-10, end=20):
    licks_during = []
    for s, e in cycles:
        l = licks - s
        licks_during.append(((l>start)*(l<end)).sum())
    return np.r_[licks_during]

def compute_lick_rate(licks, time_ax):
    lick_rate = np.zeros_like(time_ax)
    for l in licks:
        lick_rate[np.argmin(abs(l-time_ax))] += 1
    return lick_rate

def extract_single_cycle(time_ax, dff, cycles, cycle, cell,
                         cycle_start=-10):
    fc = filter_cycle(time_ax, cycles, cycle)
    t0 = time_ax[fc][0]
    return time_ax[fc] - t0 + cycle_start, dff[:, cell][fc]


def extract_single_cycle_signal(time_ax, signal, cycles, cycle, cycle_start=-10):
    fc = filter_cycle(time_ax, cycles, cycle)
    t0 = time_ax[fc][0]
    return time_ax[fc] - t0 + cycle_start, signal[fc]


def extract_single_cycle_time_ax(time_ax, cycles, cycle_duration=4, cycle_start=-10):
    # single_cycle_time_bins = filter_cycle(time_ax, cycles, cycle).sum() 
    min_len = np.inf
    truncated = 0
    ls = []
    for i, c in enumerate(cycles):
        time_ax_single = time_ax[filter_cycle(time_ax, cycles, i)]-time_ax[filter_cycle(time_ax, cycles, i)][0]
        time_ax_single = time_ax_single[time_ax_single < cycle_duration] + cycle_start
        ls.append(len(time_ax_single))
    if len(np.unique(ls))>1:
        print "Warning: I found cycles with different time bin lengths. Plus-minus one frame is generally ok."
    return time_ax_single[:np.min(ls)]

def eliminate_cycles(cycles, which_ones):
    return np.r_[[c for i, c in cycles if i not in which_ones]]

def get_cycles_durations(cycles, time_ax):
    return zip(np.diff(cycles, 1).flatten(),
               [filter_cycle(time_ax, cycles, i).sum() for i in xrange(len(cycles))])

def get_available_strings(behavior):
    return np.unique([b[1] for b in behavior]).tolist()

def read_time_ax_xml(xmlfile, sync_to_begin=False):
    # grab time axis from the xml file

    print "I infer the time axis from:\n", xmlfile
    tree = ET.parse(xmlfile)
    root = tree.getroot()
   
    time_ax = np.r_[[child.attrib['absoluteTime']
                  for child in root.iter('Frame')]].astype(float)

    if sync_to_begin:
        time_ax -= time_ax[0]

    return time_ax

def search_cycle(behavior, cycles, event):
    event_times = parse_behavior(behavior, event)
    return [any(map(lambda t: (t>=s) and (t<e), event_times)) for s, e in zip(cycles[:, 0], cycles[:, 1])]
        
def search_events(cycles, cycle, event_times):
    cycle_start, cycle_end = cycles[cycle]
    return np.r_[(event_times>=cycle_start) * (event_times<cycle_end)]

def compute_mean_level(time_ax, dff, cycles, cycle, cell, start, end, cycle_start=-10):
    t, tr = extract_single_cycle(time_ax, dff, cycles, cycle, cell, cycle_start=-10)
    return tr[(t>=start) * (t<end)].mean()

def zscore_traces(dff):
    return StandardScaler().fit_transform(dff)

def resample_signal(original_signal, original_time_ax, new_time_ax):
    return np.interp(new_time_ax, original_signal, original_time_ax)

def event_detection_cnmfe_denoised(C):
    return np.r_[[np.clip(np.r_[0, np.diff(C[:, cell])], 0, np.inf) for cell in xrange(C.shape[1])]].T

def events_to_array(events, time_ax):
    array = np.zeros_like(time_ax)
    for e in events:
        array[np.argmin(abs(time_ax-e))] += 1
    return array

def convert_p_in_stars(values, significances=[0.05, 0.01, 0.001]):
    return np.r_[["***" if r<significances[2] else
                  "**" if r<significances[1] else
                  "*" if r<significances[0] else
                  None
                 for r in values]]

def combine_cycles(time_ax, dff, cycles, cell, lim_len=-1):
    max_cycles = len(cycles)
    return np.r_[[dff[:, cell-1][ut.filter_cycle(time_ax, cycles, cycle)][:lim_len]
                  for cycle in range(max_cycles)]]

def time_to_first_event_in_cycle(cycles, behavior, event="REWARD", which_cycles=None):
    """
    Use which_cycles (a boolean list) to apply the function only to some cycles,
    for example:
        rewards = np.r_[ut.parse_behavior(behavior, 'REWARD')]
        is_rewarded = [any(map(lambda r: (r<e)*(r>=s), rewards))
                       for s, e in cycles]
        time_to_first_event_in_cycle(cycles, "REWARD", is_rewarded)
    """
    if which_cycles is None:
        which_cycles = np.r_[[True] * len(cycles)]
    event_times = np.r_[parse_behavior(behavior, event)]
#     first_event_times = np.r_[[event_times[np.where((event_times-s)>0)[0][0]]-s + CYCLE_START
#                                for s, e in cycles[np.where(which_cycles)]]]
    first_event_times = []
    for s, e in cycles[np.where(which_cycles)]:
        try:
            w = np.where((event_times-s)>0)[0][0]
            w_time = event_times[w]-s + CYCLE_START
            if w_time < CYCLE_DURATION:
                first_event_times.append(event_times[w]-s + CYCLE_START)
            else:
                first_event_times.append(-1)
        except:
            first_event_times.append(-1)
    return np.r_[first_event_times]

def shift_cycles(cycles, shifts):
    return cycles[shifts > 0] - shifts[shifts > 0][:, None]


def combine_cells(patterns_list, labels_list, train_size=0.8, patterns_per_label=100):
    all_data = [train_test_split(p, l, train_size=train_size, stratify=l)
                for p, l in zip(patterns_list, labels_list)]

    dp_training = []
    dl_training = []
    dp_test = []
    dl_test = []
    for label in np.unique(labels_list[0]):
        for i in xrange(patterns_per_label):
            choices = [np.random.choice(np.where(l==label)[0]) for p, P, l, L in all_data]
            dp_training.append(np.concatenate([p[c] for (p, P, l, L), c in zip(all_data, choices)]))
            dl_training.append(label)
        for i in xrange(patterns_per_label):
            choices = [np.random.choice(np.where(L==label)[0]) for p, P, l, L in all_data]
            dp_test.append(np.concatenate([P[c] for (p, P, l, L), c in zip(all_data, choices)]))
            dl_test.append(label)

    dp_training = np.r_[dp_training]
    dl_training = np.r_[dl_training]
    dp_test = np.r_[dp_test]
    dl_test = np.r_[dl_test]
    
    return dp_training, dp_test, dl_training, dl_test


def combine_patterns(patterns, labels, n_patterns=100, classes=[0, 1], labels_mask=None):
    labels_comb = np.r_[list(classes)*n_patterns]
    patterns_comb_train = []
    for i in xrange(n_patterns):
        for odor in classes:
            p = np.concatenate([patterns[ani][np.random.choice([w for w in np.where(labels[ani]==odor)[0]])]
                                for ani in patterns.keys()])
            patterns_comb_train.append(p)
    return np.r_[patterns_comb_train], labels_comb


def decode(decoder, patterns, labels, which_cells=None, n_loops=10, n_jobs=1, cv=10):

    if which_cells is None:
        which_cells = [True] * patterns.shape[1]
    
    scores = []
    scores_chance = []

    ps = patterns[:, which_cells]
    ls = labels
    scores = cross_val_score(decoder, ps, ls, cv=cv, n_jobs=n_jobs)
    scores_chance = []
    for i in xrange(n_loops):
        scores_chance.append(cross_val_score(decoder, ps, np.random.permutation(ls), cv=cv, n_jobs=n_jobs))
    scores_chance = np.r_[scores_chance].flatten()
    return scores, scores_chance


def load_spatial_footprints(coor_file, cnn_file=None, key='Sources2D'):
    mean_image = np.loadtxt(cnn_file) if cnn_file is not None else None
    contours = loadmat(coor_file)[key][:, 0]
    return mean_image, contours

def load_spatial_footprints_A(A_file, shape=(512, 512)):
    return np.loadtxt(A_file).T.reshape([-1, shape[0], shape[1]])


def compute_selectivity(time_ax, activity, cycles, timeframe, baseline_timeframe, stat_func=None, **stat_func_args):
    """
    timeframe and baseline_timeframe can be (START, STOP) times with respect to cycle start
    or a list of (START, STOP) times for each cycle.
    """
        
    labels_time_ax = np.zeros_like(time_ax)
    
    if timeframe is None:
        for s, e in cycles:
            labels_time_ax[(time_ax>=s) * (time_ax<e)] = 1
        
    elif len(timeframe) == 2:   
        for i, (s, e) in enumerate(cycles):
            labels_time_ax[(time_ax>=(s+timeframe[0])) *
                           (time_ax<(s+timeframe[1]))] = 1
    elif len(timeframe) == len(cycles):
        for i, ((s, e), t) in enumerate(zip(cycles, timeframe)):
            labels_time_ax[(time_ax>=(s+t[0])) *
                           (time_ax<(s+t[1]))] = 1
    else:
        raise Exception('Length of timeframe is %d and len of cycles is %d. Should be the same instead'%
                        (len(timeframe), len(cycles)))

    if len(baseline_timeframe) == 2:   
        for i, (s, e) in enumerate(cycles):
            labels_time_ax[(time_ax>=(s+baseline_timeframe[0])) *
                           (time_ax<(s+baseline_timeframe[1]))] = -1
    elif len(baseline_timeframe) == len(cycles):
        for i, ((s, e), t) in enumerate(zip(cycles, baseline_timeframe)):
            labels_time_ax[(time_ax>=(s+t[0])) *
                           (time_ax<(s+t[1]))] = -1
    else:
        raise Exception('Length of timeframe is %d and len of cycles is %d. Should be the same instead'%
                        (len(baseline_timeframe), len(cycles)))
        
    if stat_func is None:
        stat_func = lambda x, y: sstats.mannwhitneyu(x, y, alternative='two-sided')

    selectivity = []
    for cell in xrange(activity.shape[1]):
        act_cs = activity[:, cell][labels_time_ax==1]
        act_base = activity[:, cell][labels_time_ax==-1]
        try:
            selectivity.append([stat_func(act_cs, act_base, **stat_func_args),
                                np.sign(np.mean(act_cs)-np.mean(act_base))])
        except ValueError:
            # if activity is 0 in both conditions
            selectivity.append([np.r_[0, 1], 0])

    return np.r_[selectivity]



def adjust_pvalues(pvalues, method='fdr_bh', **method_args):
    return multipletests(pvalues, method=method, **method_args)[1]


def extract_patterns(time_ax, activity, cycles, CYCLE_START, STIM_START, STIM_END, mode='average'):
    n_cells = activity.shape[1]
    if mode == 'average':
        patterns = np.zeros((len(cycles), n_cells))
    elif mode == 'corr':
        cov_model = EmpiricalCovariance()
        patterns = np.zeros((len(cycles), n_cells, n_cells))
    # tas is a replacement for time_ax_single
    for i, (s, e) in enumerate(cycles):
        time_filter = ((time_ax>=(s-CYCLE_START+STIM_START)) * (time_ax<(s-CYCLE_START+STIM_END)))
        if mode == 'average':
            patterns[i] = activity[time_filter].mean(0)
        elif mode == 'corr':
            patterns[i] = cov_model.fit(activity[time_filter]).covariance_
    patterns = patterns.reshape(len(cycles), -1)
    # patterns = patterns[:, which_cells]
    
    return patterns


def decode(patterns, labels, decoder=None, which_cells=None, n_loops=10, n_jobs=1, cv=10):

    if which_cells is None:
        which_cells = [True] * patterns.shape[1]
    
    if decoder is None:
        decoder = SVC(kernel='linear')

    scores = []
    scores_chance = []

    ps = patterns[:, which_cells]
    ls = labels
    scores = cross_val_score(decoder, ps, ls, cv=cv, n_jobs=n_jobs)
    scores_chance = []
    for i in xrange(n_loops):
        scores_chance.append(cross_val_score(decoder, ps, np.random.permutation(ls), cv=cv, n_jobs=n_jobs))
    scores_chance = np.r_[scores_chance].flatten()
    return scores, scores_chance


def compute_similarity_matrix(pattern_ids, all_patterns, similarity_func=None):
    if similarity_func is None:
        similarity_func = lambda x, y: sstats.pearsonr(x, y)[0]
    corrmat_distr = {}
    for i, (l, a) in enumerate(zip(pattern_ids, all_patterns)):
        for j, (m, b) in enumerate(zip(pattern_ids, all_patterns)):
            temp = []
            try:
                for ii, aa in enumerate(a):
                    for jj, bb in enumerate(b):
                        if ii==jj or aa.sum()==0 or bb.sum()==0: continue
                        temp.append(similarity_func(aa, bb))
            except ValueError:
                print "Cannot compute similarity between %s and %s." % (l, m)
            corrmat_distr[(l, m)] = temp
    corrmat = np.zeros((len(pattern_ids), len(pattern_ids)))
    for i, p in enumerate(pattern_ids):
        for j, q in enumerate(pattern_ids):
            corrmat[i][j] = np.mean(corrmat_distr[(p, q)])
    return corrmat_distr, corrmat
            

def compute_mean_activity_patterns(time_ax, activity, cycles, timeframe):
    start, stop = timeframe
    return np.r_[[np.mean(activity[(time_ax >= (s+start)) * (time_ax < (s+stop))], 0)
                 for s, e in cycles]]



def compute_similarity_matrix_woods(pattern_ids, all_patterns):
    
    corrmat_distr = {}
    for i, (l, a) in enumerate(zip(pattern_ids, all_patterns)):
        for j, (m, b) in enumerate(zip(pattern_ids, all_patterns)):
            temp = []
            for ii, aa in enumerate(a):
                for jj, bb in enumerate(b):
                    # skip if same vector or any of the 2 is zero
                    if ii==jj or np.sum(aa)==0 or np.sum(bb)==0: continue
                    # count once if using same data
                    if l==m and jj<ii: continue
                    temp.append(sstats.pearsonr(aa, bb)[0])
            corrmat_distr[(l, m)] = temp
    corrmat = np.zeros((len(pattern_ids), len(pattern_ids)))
    for i, p in enumerate(pattern_ids):
        for j, q in enumerate(pattern_ids):
            corrmat[i][j] = np.mean(corrmat_distr[(p, q)])

    return corrmat_distr, corrmat

           
def extract_traces_around_event(time_ax, traces, evs, tpre, tpost):
    extracted = [traces[(time_ax>=(e-tpre))*(time_ax<(e+tpost))]
                        for e in evs]
    min_len = np.min([len(t) for t in extracted])
    return np.r_[[t[:min_len] for t in extracted]]


def extract_activity(time_ax, activity, cycles, CYCLE_START, STIM_START, STIM_END,
                     offset=0, which=None):
    if which is None:
        which = [True] * len(cycles)
    return np.r_[[activity[(time_ax>=(start-CYCLE_START+STIM_START+offset))
                           *(time_ax<(start-CYCLE_START+STIM_END+offset))].mean(0)
                  for start, stop in cycles[which]]]


def generate_combined_cells_patterns(patterns_dict, labels_dict, n_patterns=30, labels=None, animals=None):
    if labels is None:
        labels = np.unique(np.concatenate(labels_dict.values()))
    if animals is None:
        animals = patterns_dict.keys()
    patterns_combined = []
    for o in labels:
        patterns_o = []
        for m, v in patterns_dict.iteritems():
            patterns_o.append(np.c_[[np.random.choice(v[labels_dict[m]==o][:, cell], size=n_patterns)
                                     for cell in range(v.shape[1])]].T)
        patterns_combined.append(np.column_stack(patterns_o))
    patterns_combined = np.row_stack(patterns_combined)

    labels_combined = np.r_[[[o]*n_patterns for o in labels]].flatten()
    
    return patterns_combined, labels_combined


def sig_95(vals):
    return [0, 0 if ((np.mean(vals)-50)/(sstats.sem(vals)*2))>=1 else 1]

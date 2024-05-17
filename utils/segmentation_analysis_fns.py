# Imports 
from glob import glob
import re
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np 
from tifffile import TiffFile, imwrite
from IPython.display import clear_output # this modules allows to flush output
import matplotlib as mpl
import copy

from matplotlib import cm
import colorcet as cc
import pickle
import itertools
from itertools import groupby
from copy import deepcopy 
from numpy import array
import pandas as pd
import plotly.graph_objects as go 
from mltools.batch_processing import open_dets

HOME = '/home/idjafc/'
mpl.rcParams['figure.dpi'] = 100
DEBUG = True

def find_peak(inp,debug=False,lab=None):
    sim_scores = array([x.get('sim_scores',None) for x in inp])
    if all(sim_scores.flatten() == None): return np.ones(len(sim_scores)+1)*False
    is_peak = np.diff(sim_scores) > +0.05 
    is_peak = np.append(is_peak,False) 
    is_peak = is_peak | (sim_scores < 0.1)
    peak_loc = np.where(is_peak)[0]
    nopeak_loc = np.where(~is_peak)[0]
    if debug:
        plt.plot(sim_scores,label=lab)
        plt.plot(peak_loc,sim_scores[peak_loc],'*');
    return is_peak

def load_recursive(OUT,regex):
    ''' Takes a folder and a regular expression and returns a dict with (well_name,grid_x,grid)'''
    EXT = '.pkl'
    all_files = OUT + '*' + EXT
    glob_files = glob(all_files)
    preds = {}
    #files_matching_regex = [re.findall(regex,fname) for fname in glob_files]
    #files_matching_regex = list(filter(files_matching_regex,)
    #all_wells = set(re.findall(regex,fname)[0][0] for fname in files_matching_regex if fname != []  )

    for fname in glob_files: 
        if re.findall(regex,fname): 
            with open(fname,'rb') as f:
                well,i,j = list(map(int,re.findall(regex,fname)[0]))
                preds[(well,i,j)] = pickle.load(f)
    return preds

            
def load_recursive_merge(OUT,regex):
    EXT = '.pkl'
    all_files = OUT + '/**/*' + EXT
    glob_files = glob(all_files,recursive=True)
    preds = {}
    all_wells = set(re.findall(regex,fname)[0][0] for fname in glob_files)
#     return glob_files
    for fname in glob_files: 
        with open(fname,'rb') as f:
            coord = list(map(int,re.findall(regex,fname)[0]))
            coord = tuple(coord)
            if preds.get(coord):
                preds[coord].append(pickle.load(f))
            else: 
                preds[coord] = [pickle.load(f)]
    return preds


def plot_counts_wtime(preds,thrs,delta_t=5):
    fig,ax = plt.subplots(2,2,figsize=(15,10))
    ax[0][0].set_title('No filtering')
    ax[0][1].set_title('Shuffled images filtered')
    x = None
    for well in range(len(preds.items())):
        # dim: nview
        dat = [v for p,v in preds.items() if p[0] == well+1 ]
        if dat:
            # dims: nview*n_timepoints
            ndet = [[np.sum(timepoint['scores']>thrs)  for timepoint in view ] for view in dat]
            masks = [ find_peak(x)  for x in dat]
            
            time = [[timepoint['time']  for timepoint in view ] for view in dat]
            # Only process views which have same number of timepoints
            if all( [len(view) == len(dat[0]) for view in dat]):
                ndet = array(ndet)
                dat_not_masked = np.sum(ndet,axis=0)
                ndet_all = dat_not_masked/dat_not_masked[0]
                
                if masks:
                    masks = array(masks)
                    assert masks.shape == ndet.shape
                    ndet_masked = ndet.copy()
                    ndet_masked[masks] = 0 

                    frac_views = 1.0 - np.sum(masks,axis=0)/ndet.shape[1]
                    dat_masked = np.sum(ndet_masked,axis=0)
                    
                    ndet_all_masked = dat_masked*frac_views/dat_masked[0]
                    
                if 'U87' in labels[well]:
                    ls = '--'
                else:
                    ls = '-'
    #             timepoints = np.arange(len(ndet_all))*(delta_t/60)

                # Convert absolute time in relative one (ms)
                timepoints = trim_timestamp(time).astype('datetime64')
                timepoints = (timepoints-timepoints[:,0][:,None])/np.timedelta64(1,'h')            
                timepoints = np.average(timepoints,axis=0)

                ax[0][0].plot(timepoints,ndet_all,ls=ls,label=f"{labels[well]}")
                ax[0][1].plot(timepoints,ndet_all_masked,ls=ls,label=f"{labels[well]}")

                ax[1][0].plot(timepoints[:-4],rolling_average(ndet_all,5),ls=ls,label=f"{labels[well]}")
                ax[1][1].plot(timepoints[:-4],rolling_average(ndet_all_masked,5),ls=ls,label=f"{labels[well]}")
        #    if well+1> 8:
                all_data['exp2'+labels[well]] = (timepoints,ndet_all)

    ax[0][0].legend()
    ax[0][1].legend()
    ax[1][0].legend()
    ax[1][1].legend()
    

# All the functions we need 

# This function handles cases where the number of timepoints differ 
# It returns only the averaged quantities, another function plots it 
def counts_wtime(preds,labels,thrs,exp_label='',min_idx=1):
    from itertools import zip_longest
    all_wells = sorted(set(k[0] for k in preds.keys()))
    all_data_loc = {}
    assert len(all_wells) == len(labels), print(list(zip_longest(labels,all_wells,))) 
    for label,well in zip(labels,all_wells): 
        if DEBUG == True: print(f'{label}--->{well}')
        # dim: nfolder
        dat = [v for p,v in preds.items() if p[0] == well ]
        if dat:
            #print(f'{well} {label}')
            # dims: nview*n_timepoints
            n = min([len(view) for view in dat])
            #print(label,n)
            # Trim excess timepoints
            dat = [ view[:n] for view in dat]
            ndet = [[np.sum(timepoint['scores']>thrs)  for timepoint in view ] for view in dat]
            masks = [ find_peak(x)  for x in dat]
            time = [[timepoint['time']  for timepoint in view ] for view in dat]
            #assert len(set( [len(view) for view in dat])) == 1, [n,set( [len(view) for view in dat])]

            ndet = array(ndet)
            dat_not_masked = np.sum(ndet,axis=0)
            ndet_all = dat_not_masked #/dat_not_masked[0]

            # Try to remove images which are shuffled
            if all( any(m != False) for m in masks):
                masks = array(masks)
                assert masks.shape == ndet.shape
                ndet_masked = ndet.copy()
                ndet_masked[masks] = 0 
                frac_views = 1.0 - np.sum(masks,axis=0)/ndet.shape[1]
                dat_masked = np.sum(ndet_masked,axis=0)
                ndet_all_masked = dat_masked*frac_views/dat_masked[0]
                
            
            # Convert absolute time in relative one (ms)
            epoch = np.datetime64('1970-01-01T00:00:00')
            timepoints = trim_timestamp(time).astype('datetime64')
            timepoints = (timepoints-epoch)/np.timedelta64(1,'h')            
            timepoints = np.average(timepoints,axis=0)
            # We return averaged quantities, this is maybe not the smartest idea ... 
            all_data_loc[exp_label+label] = [timepoints,ndet]
    return all_data_loc

def equalize_batch(preds):
    all_wells = set(k[0] for k in preds.keys())
    
    n = {}
    n_max = {}
    for well in all_wells: 
        # dim: nfolder
        dat = [v for p,v in preds.items() if p[0] == well ]
        n[well] = min([len(x) for x in dat])
        n_max[well] = max([len(x) for x in dat])
        #print(f'{well}:{[len(x) for x in dat]}')
        #print(f'well {well} has between {n[well]} elements and {n_max[well]} elements]')
    preds = { k:v[:n[k[0]] ] for k,v in preds.items()}
    return preds 

def load_all_folders(ROOT,labels,exp_label,start=0,zero=0,thrs=0.9,
                     regex = 'MMStack_(.*?)-Pos(.*?)_(.*)\.',normalize=True,average=True,min_idx=1):
    ''' 
    Takes an interrupted experiment split into several folders 
    and merge it into a single dataset
    '''
    counts_all_dir = {}
    if type(ROOT) == str:
        folders = sorted(glob(os.path.join(ROOT,'*/'), recursive = True))[start:]
    else: 
        folders = ROOT
        
    for folder in folders:#     for folder in sorted(glob(os.path.join(ROOT,'*/'), recursive = True))[start:]:
        print(folder)
        preds = load_recursive(folder,regex)
        preds = equalize_batch(preds)
        counts = counts_wtime(preds,labels,thrs,
                                      exp_label=exp_label,min_idx=min_idx)

        for well,v in counts.items():
            if average == True:
                counts[well][-1] = np.average(counts[well][-1],axis=0)
            else: 
                counts[well][-1] = np.sum(counts[well][-1],axis=0)
            
        for well,v in counts.items():
            if counts_all_dir.get(well):
                counts_all_dir[well]+= v 
            else: 
                counts_all_dir[well] = v
    for k in counts_all_dir:
        counts_all_dir[k] = [arr[zero:] for arr in counts_all_dir[k] ]
    
    counts_all_dir = remove_min_time(counts_all_dir)
    if normalize:
        counts_all_dir = normalize_ndet(counts_all_dir)
    return counts_all_dir

def remove_min_time(counts_all_dir):
    debug=False
    min_time = min([min(v[0]) for k,v in counts_all_dir.items()])
    if debug: print(min_time)
    for well,v in counts_all_dir.items():
        for time in counts_all_dir[well][0::2]:
            time -= min_time
    return counts_all_dir

def normalize_ndet(counts_all_dir,zero=0):
    for well in counts_all_dir:
        ndet0 = copy.deepcopy(counts_all_dir[well][1][zero])
        for ndet in counts_all_dir[well][1::2]:
            ndet /= ndet0
    return counts_all_dir


def plot(counts_all_dir,roll=1,labs=['Hela','U87MG'],relative=True):
    #dat = copy.deepcopy(counts_all_dir)
    dat = {}
    n = len(labs)
    fig,axs = plt.subplots(1,n,figsize=(20,10),squeeze=True)
    #for ax in axs.flat: ax.set_box_aspect(1)
    for j,(well,v) in enumerate(counts_all_dir.items()):
        well_matches_lab = [ re.findall(lab,well) != [] for lab in labs]
        try:
            ax_id = well_matches_lab.index(True)
        except Exception as e:
            print(f'skipping {well}')
            continue
        ax = axs[ax_id] #if lab_left in well else axs[1]
        dat[well] = [rolling_average(x,roll) for x in v]
        v = dat[well]
        if relative == True:
            # 'normalize' by the value at t=1
            v[1::2] /=(v[1][0])
        ax.plot(*v[:2],label=well,color=cc.cm.glasbey(j)) # one label
        ax.plot(*v[2:],color=cc.cm.glasbey(j)) #x,y,x,y,x,y
    [ax.legend() for ax in axs]
    return dat 
    
def get_growth_rate(x,time,N):
    growth_rate = np.log(x[N:]/x[:-N])/(time[N:]-time[:-N])
    time = time[:-N]
    return time,growth_rate

def plot_growth_rate(counts_all_dir,N=5,roll=1):
    dat = copy.deepcopy(counts_all_dir)
    fig,axs = plt.subplots(1,2,figsize=(15,10),squeeze=True)
    for j,(well,v) in enumerate(counts_all_dir.items()):
        ax = axs[0] if 'Hela' in well else axs[1]
        v = [rolling_average(x,roll) for x in v]
        time,growth_rate = get_growth_rate(v[1],v[0],N)
        dat[well] = [time,growth_rate]
        ax.plot(time,growth_rate,label=well,color=cc.cm.glasbey(j))
    [ax.legend() for ax in axs]    
    return dat 


def sort_labels(x): 
    conc = x.split(' ')[1]
    try:
         return float(conc)
    except:
        return -1 
    
#def compare_exp(dat1,tag1,dat2,tag2):
def compare_exp(dats,tags):
    assert len(dats) == len(tags)
    all_labels = [x.removeprefix(tag) for dat,tag in zip(dats,tags) for x in dat.keys()]
    for celltype in ['Hela','U87MG']:
        all_labels_celltype = set([ lab for lab in all_labels if celltype in lab])
        all_labels_celltype = sorted(all_labels_celltype,key=sort_labels)
        print(all_labels_celltype)
        for lab in all_labels_celltype:
        #    if lab in labels2:
            fig,ax = plt.subplots(1,2,figsize=(10,5))
            ax[0].set_title(lab+' fold change')
            ax[1].set_title(lab+' normalised by control')

            #celltype = 'Hela' if 'Hela' in lab else 'U87MG'
            for dat,tag,col in zip(dats,tags,['red','green','blue','pink']): 
                lab_well = tag+lab
                dat_exp = dat.get(tag+lab,None)
                if dat_exp:
                    # plot 
                    ax[0].plot(*dat_exp[:2],label=tag,color=col)
                    ax[0].plot(*dat_exp[2:],color=col)                    
                    # plot control
                    label_control = tag+celltype+' control'
                    control = dat[label_control]
                    #ax[0].plot(*control[:2],color='black',label='control '+tag,)
                    #ax[0].plot(*control[2:],color='black',)
                    ax[0].legend()
                    
                    # normalize vs control 
                    ndet = copy.deepcopy(dat_exp[1::2])
                    ndet_normed = [x[:min([len(x),len(c)])]/c[:min([len(x),len(c)])] for x,c in zip(dat_exp[1::2],control[1::2])]
                    dat_exp_normed = list( y for x in zip(dat_exp[0::2],ndet_normed) for y in x)
                    ax[1].plot(*dat_exp_normed[:2],label=tag,color=col)
                    ax[1].plot(*dat_exp_normed[2:],color=col)
                    ax[1].legend()
                    
def dic_to_3D_dataframe(x,cols):
    ''' takes a dict and makes it a "3D" dataframe by using the well label 
        as heading and column names as subheading '''
    x = concat_different_series(x)

    return {(k,name):val for k,v in x.items() for val,name in zip(v,cols)}

def normalize_by_control(arr,control_label):
    ''' Takes the output dict and normalize the values by the control (which is given by the control label)'''
    tmp = copy.deepcopy(arr)
    vref = copy.deepcopy(tmp[control_label][1::2])
    for k,v in tmp.items():
        for vals,ref in zip(v[1::2],vref):
            vals/= ref
    return tmp

def concat_different_series(dic):
    ''' Takes the output dict and normalize the values by the control (which is given by the control label)'''
    return {k:[np.concatenate(v[0::2]),np.concatenate(v[1::2])] for k,v in dic.items()}

def get_sub_dict(dic,string):
    ''' get a sub dict given by the key '''
    return { k:v for k,v in dic.items() if string in k }

        
# This function handles cases where the number of timepoints differ 
def plot_counts_wtime2(preds,labels,thrs,delta_t=5,exp_label=''):
    thrs = 0.5
    delta_t = 5 #time between timepoints in minute
    fig,ax = plt.subplots(2,2,figsize=(15,10))
    ax[0][0].set_title('No filtering')
    ax[0][1].set_title('Shuffled images filtered')
    x = None
    all_wells = set(k[0] for k in preds.keys())
    all_data_loc = {}

    for well in all_wells: #range(len(preds.items())):
        # dim: nfolder
        dat = [v for p,v in preds.items() if p[0] == well ]
        
        # Trim excess timepoints
        n = [[len(x) for x in folder] for x in dat]
        t = [ ]
        
        if dat:
            label = labels[well-1] #wells are indexed starting from 1 
            # dims: nview*n_timepoints
            ndet = [[np.sum(timepoint['scores']>thrs)  for timepoint in view ] for view in dat]
            masks = [ find_peak(x)  for x in dat]
            time = [[timepoint['time']  for timepoint in view ] for view in dat]
            # Trim excess timepoints
            for x in ['time','masks','ndet']:
                locals()[x] = [ y[:n] for y in x ]
            #assert len(set( [len(view) for view in dat])) == 1, [n,set( [len(view) for view in dat])]
            if all( [len(view) == len(dat[0]) for view in dat]):
                ndet = array(ndet)

                # Try to remove images which are shuffled
                masks = array(masks)
                assert masks.shape == ndet.shape
                ndet_masked = ndet.copy()
                ndet_masked[masks] = 0 
                frac_views = 1.0 - np.sum(masks,axis=0)/ndet.shape[1]
                dat_masked = np.sum(ndet_masked,axis=0)
                dat_not_masked = np.sum(ndet,axis=0)
                ndet_all = dat_not_masked/dat_not_masked[0]
                ndet_all_masked = dat_masked*frac_views/dat_masked[0]

                # Convert absolute time in relative one (ms)
                epoch = np.datetime64('1970-01-01T00:00:00')
                timepoints = trim_timestamp(time).astype('datetime64')
                timepoints = (timepoints-epoch)/np.timedelta64(1,'h')            
                timepoints = np.average(timepoints,axis=0)

    #                 ax[0][0].plot(timepoints,ndet_all,ls=ls,label=f"{label}")
    #                 ax[0][1].plot(timepoints,ndet_all_masked,ls=ls,label=f"{label}")

    #                 ax[1][0].plot(timepoints[:-4],rolling_average(ndet_all,5),ls=ls,label=f"{label}")
    #                 ax[1][1].plot(timepoints[:-4],rolling_average(ndet_all_masked,5),ls=ls,label=f"{label}")
        #    if well+1> 8:
    #                dat[label] = (timepoints,ndet_all)
                all_data_loc[label] = [timepoints,ndet_all]
        
    min_time = min([ min(time)for time,_ in all_data_loc.values()])
    
    for label,(timepoints,ndet_all) in all_data_loc.items():
        if 'U87' in label:
            ls = '--'
        else:
            ls = '-'
            
        #all_data_loc[label][0] = timepoints - min_time 
        # Offset data 
        timepoints = timepoints - min_time
        ax[0][0].plot(timepoints,ndet_all,ls=ls,label=f"{label}")
        #ax[0][1].plot(timepoints,ndet_all_masked,ls=ls,label=f"{label}")

        ax[1][0].plot(timepoints[:-4],rolling_average(ndet_all,5),ls=ls,label=f"{label}")
        #ax[1][1].plot(timepoints[:-4],rolling_average(ndet_all_masked,5),ls=ls,label=f"{label}")
        
        all_data[exp_label+label] = [timepoints,ndet_all]

        
#     for k,v in dat: 
#         min = 
    
#     min_time = min([ min(l.get_xdata()) for l in ax[0][0].lines])

#     for l in ax[0][0].lines: 
#         dat = l.get_xdata()
#         l.set_xdata(dat - min_time)
#     mintime = np.min([l for l in ax[0][0].lines])

    ax[0][0].legend()
    ax[0][1].legend()
    ax[1][0].legend()
    ax[1][1].legend()

    
def plot_and_export(preds,labels,thrs=0.9,key=None):

    fig,ax = plt.subplots(1,2,figsize=(15,10))
    ret = {}
    for well in range(16):
        dat = {(p[1],p[2]):v for p,v in preds.items() if p[0] == well+1 }
        dat = [ [np.sum(x>thrs) for x in g] for g in dat.values() ]
        dat = np.sum(np.array(dat),axis=0)
        label = labels[well]
        if well+1> 8:
            ax[0].plot(dat/dat[0],'--',label=f"well={label}")
        else:
            ax[1].plot(dat/dat[0],label=f"well={label}")
        timepoints = np.arange(len(dat))
        ret[key+label] = (timepoints,dat/dat[0])
    ax[0].legend();
    ax[1].legend();
    return ret 

def rolling_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# Function that lowers the precision of the timestamp (after the seconds)
trim_timestamp = np.vectorize(lambda x: x[:-6])


def find_score_threshold_folder(ROOT,regex = 'MMStack_(.*?)-Pos(.*?)_(.*)\.'):
    ''' 
        Takes a folder made of detected objects and finds the lowest detection 
        confidence object(should be ~ the score threshold)
    '''
    # As I forgot what score threshold I set, I need to find it one way or another.
    counts_all_dir = {}
    for folder in sorted(glob(os.path.join(ROOT,'*/'), recursive = True)):
        preds = load_recursive(folder,regex)
        #print(folder, [min([x['scores'] for x in view]) for k,view in preds.items()])
        minv = 1.0
        for k,view in preds.items():
            x = [timepoint['scores'] for timepoint in view] 
            x = [ min(y)  for y in x if len(y)>0]
            if len(x)  > 0 :
                minv = min(min(x),minv)
        print(folder,minv)









def load_all_folders_pandas(ROOT,labels,exp_label,start=0,zero=0,thrs=0.9,
                     regex = 'MMStack_(.*?)-Pos(.*?)_(.*)\.',normalize=True,average=True,min_idx=1):
    ''' 
    Takes an interrupted experiment split into several folders 
    and merge it into a single dataset
    '''
    counts_all_dir = {}
    if type(ROOT) == str:
        folders = sorted(glob(os.path.join(ROOT,'*/'), recursive = True))[start:]
    else: 
        folders = ROOT
        
    for folder in folders:#     for folder in sorted(glob(os.path.join(ROOT,'*/'), recursive = True))[start:]:
        print(folder)
        preds = load_recursive(folder,regex)
        preds = equalize_batch(preds)
        df = counts_wtime_pandas(preds,labels,thrs,
                                      exp_label=exp_label,min_idx=min_idx)
        
        

        sliceof = df.loc[:,(slice(None),'time'),]
        absmin = sliceof.to_numpy().min()
    
        df.loc[:,(slice(None),'relative_time'),] = sliceof - absmin
        sliceof = df.loc[:,(slice(None),'time'),]
        absmin = sliceof.to_numpy().min()
        sliceof.rename(columns={'time':'relative_time'},inplace=True)
        sliceof -= absmin
        df = pd.concat([df,sliceof],axis=1)

        for well,v in counts.items():
            if average == True:
                counts[well][-1] = np.average(counts[well][-1],axis=0)
            else: 
                counts[well][-1] = np.sum(counts[well][-1],axis=0)
            
        for well,v in counts.items():
            if counts_all_dir.get(well):
                counts_all_dir[well]+= v 
            else: 
                counts_all_dir[well] = v
    for k in counts_all_dir:
        counts_all_dir[k] = [arr[zero:] for arr in counts_all_dir[k] ]
    
    counts_all_dir = remove_min_time(counts_all_dir)
    if normalize:
        counts_all_dir = normalize_ndet(counts_all_dir)
    return counts_all_dir



def save_results_to_file(dic,filename_no_ext):
    assert not '.' in filename_no_ext
    rel_cnts = plot(dic,roll=1,labs=['HeLa','U87MG'],relative=True)
    df = pd.DataFrame(dic_to_3D_dataframe(rel_cnts,cols=['time','#cell(t)/#cell(t=0)']))
    df.to_excel(f'{filename_no_ext}_relative_number_of_cells.xlsx')
    df.to_pickle(f'{filename_no_ext}_relative_number_of_cells.pkl')

    abs_cnts = plot(dic,roll=1,labs=['HeLa','U87MG'],relative=False)
    df = pd.DataFrame(dic_to_3D_dataframe(abs_cnts,cols=['time','#cell(t)']))
    df.to_excel(f'{filename_no_ext}_absolute_number_of_cells.xlsx')
    df.to_pickle(f'{filename_no_ext}_absolute_number_of_cells.pkl')





def counts_wtime_pandas(preds,labels,thrs,exp_label='',min_idx=1):

    all_wells = sorted(set(k[0] for k in preds.keys()))
    all_data_loc = []
    assert len(all_wells) == len(labels)
    for label,well in zip(labels,all_wells): 
        if DEBUG == True: print(f'{label}--->{well}')
        # dim: nfolder
        dat = [v for p,v in preds.items() if p[0] == well ]
        if dat:
            #print(f'{well} {label}')
            # dims: nview*n_timepoints
            n = min([len(view) for view in dat])
            #print(label,n)
            # Trim excess timepoints
            dat = [ view[:n] for view in dat]
            ndet = [[np.sum(timepoint['scores']>thrs)  for timepoint in view ] for view in dat]
            masks = [ find_peak(x)  for x in dat]
            time = [[timepoint['time']  for timepoint in view ] for view in dat]
            #assert len(set( [len(view) for view in dat])) == 1, [n,set( [len(view) for view in dat])]

            ndet = array(ndet,dtype=int)
            
            dat_not_masked = np.sum(ndet,axis=0)
            ndet_all = dat_not_masked #/dat_not_masked[0]

            # Try to remove images which are shuffled
            if all( any(m != False) for m in masks):
                masks = array(masks)
                assert masks.shape == ndet.shape
                ndet_masked = ndet.copy()
                ndet_masked[masks] = 0 
                frac_views = 1.0 - np.sum(masks,axis=0)/ndet.shape[1]
                dat_masked = np.sum(ndet_masked,axis=0)
                ndet_all_masked = dat_masked*frac_views/dat_masked[0]
                
            # Convert absolute time in relative one (ms)
            epoch = np.datetime64('1970-01-01T00:00:00')
            timepoints = trim_timestamp(time).astype('datetime64')
            timepoints = (timepoints-epoch)/np.timedelta64(1,'h')            
            timepoints = np.average(timepoints,axis=0)
            
            #ndet_avg = pd.Series(ndet.mean(axis=0),name='avg_num_cells')
            # ndet = pd.Series(ndet.sum(axis=0),name='num_cells')
            # timepoints = pd.Series(timepoints,name='time')

    all_data_loc = pd.concat(all_data_loc,keys=labels,axis=1)
    return all_data_loc

def relative_counts(counts,relative_timepoint):
    counts_relative = copy.deepcopy(counts)
    for j,(well,v) in enumerate(counts.items()):
        counts_relative[well][1::2]/=counts_relative[well][1][relative_timepoint]
    return counts_relative

def plot_plotly(counts_all_dir,roll=1,relative_timepoint=0,labs=None,relative=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if relative:
        counts_all_dir = relative_counts(counts_all_dir,relative_timepoint)

    fig = go.Figure()
    fig = make_subplots(rows=1, cols=len(labs))
    dat = {}
    n = len(labs)

    for j,(well,v) in enumerate(counts_all_dir.items()):
        well_matches_lab = [ re.findall(lab,well) != [] for lab in labs]
        try:
            ax_id = well_matches_lab.index(True)
        except Exception as e:
            print(f'skipping {well}')
            continue
        #ax = axs[ax_id] #if lab_left in well else axs[1]
        
        dat[well] = [rolling_average(x,roll) for x in v]
        v = dat[well]
        # if relative == True:
        #     # 'normalize' by the value at t=1
        #     v[1::2] /=(v[1][0])

        # x = *v[:2]
        # y = *v[:2]

        x = np.concatenate(v[0::2])
        y = np.concatenate(v[1::2])
        fig.add_trace(go.Scatter(x=x, y=y,
                    mode='lines', name=well,text=well),row=1,col=ax_id+1)
        # plot(*v[:2],label=well,color=cc.cm.glasbey(j)) # one label
        # for x,y in v[2::2]:
        #     fig.add_trace(go.Scatter(x=v[0], y=v[1], mode='lines', color = cc.cm.glasbey(j), name=well))
        
    #[ax.legend() for ax in axs]
    fig.update_layout(
        #autosize=True
        width = len(labs)*1000,
        height = 1000,
    )

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        font=dict(
    #        family="Courier New, monospace",
            size=30,
    #        color="RebeccaPurple"
        )
    )

    fig.update_traces(line=dict(width=5.0))

    #fig.show()
    return fig


# Handling bin files originating from nd2


def parse_dat_from_nd2(files,labels=None,thrs=0.5,delta_time=None):
    thrs = 0.5
    #files = glob(globexp)
    res = {}
    
    if labels: 
        assert len(labels) == len (files)
    else:
        labels = [Path(f).stem for f in files]
        
    for f,lab in zip(files,labels):
        print(f+"---->"+lab)
        dets = open_dets(f)
        counts = [len(det.instances_all.scores > thrs) for det in dets]
        if hasattr(dets[0].metadata,'channels'):
            time = [det.metadata.channels[0].time.relativeTimeMs for det in dets]
        else: 
            assert delta_time != None, Warning('Cannot find time data in the file, please provide `delta_time` variable ')
            if hasattr(dets[0],'coords'):
                time = delta_time*np.array([det.coords['t'] for det in dets])
            else:
                Warning('Cannot find timepoint coordinates in the file, assuming images are sorted and contiguous')
                time = np.arange(len(dets))*delta_time
        res[lab] = pd.DataFrame.from_dict({'counts':np.array(counts),'time':pd.to_timedelta(time,unit='ms')})
        
    df = pd.concat(res.values(),keys=res.keys(),axis=1)
    return df 
    

# Plotting functions
def plot_dat_from_nd2(df,norm=True):
    for lab,dat in df.iteritems():
        counts = dat.counts_normalized if norm else dat.counts
        time = dat.time / np.timedelta64(1,'h')
        plt.plot(time,counts,label=lab)
    plt.xlabel('time (h)')
    plt.ylabel('relative counts' if norm else 'counts')
    plt.legend()
    
def plot_df_plotly(counts_all_dir,roll=1,labs=[],relative=True,diff=False):
    from plotly.subplots import make_subplots
    n = max(len(labs),1)
    tags = counts_all_dir.columns.get_level_values(0).drop_duplicates()
    fig = make_subplots(rows=1, cols=n)

    for well in tags:
        dat = counts_all_dir[well].copy()
        dat = dat.set_index('time')
        #x = dat.time/np.timedelta64(1,'h') # time in hours
        y = dat.counts
        
        if roll > 1:    
            y = y.rolling(roll).mean().dropna()
    
        if type(relative) is bool and relative is True: 
            y = y/y[0]
        elif type(relative) == str :            
            y = y/y[0]
            ref = counts_all_dir[relative].copy()
            ref = ref.set_index('time')
            ref = ref.counts
            if roll > 1:
                ref = ref.rolling(roll).mean().dropna()
            ref = ref/ref[0]
            y = y/ref.values
            #return y,ref
        # if well != '01_Control' :return y
        # else: 
        #     continue
        from functools import partial

        if diff:
            grad = pd.Series(np.gradient(y),index=y.index)
            y.update(grad)
            #y['values'] = partial(lambda x: np.gradient(x.values,x.index))(y)
        
        fig.add_trace(go.Scatter(x=y.index/np.timedelta64(1,'h'),y=y,
                    mode='lines', name=well,text=well),row=1,col=1)
    
    yaxis_title = ("relative "*bool(relative)) 
    yaxis_title += "counts" + (" slope"*diff) 
    if type(relative) == str: 
        yaxis_title += f" ref: {relative}"  
    
    fig.update_layout(
        autosize=True,
        #width = 1000,
        height = 1000,
        xaxis_title="time (h)",
        yaxis_title= yaxis_title,
        legend_title="well label",
        # font=dict(
        #     family="Courier New, monospace",
        #     size=18,
        #     color="RebeccaPurple"
        # )
        )

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        font=dict(
    #        family="Courier New, monospace",
            size=30,
    #        color="RebeccaPurple"
        )
        )

    fig.update_traces(line=dict(width=5.0))
    return fig 

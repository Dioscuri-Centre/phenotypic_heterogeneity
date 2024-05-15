import os.path
import numpy as np
from matplotlib import pyplot as plt
from LineageTree import lineageTree
import scipy.stats as stats
import xml.etree.ElementTree as ET
import os
# fix for ete 
# os.environ['QT_QPA_PLATFORM']='offscreen'

from ete3 import TreeStyle,NodeStyle
plt.rcParams.update({'font.size': 22})

def cartesian(arrays, out=None):
    """
    Generate a Cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the Cartesian product of.
    out : ndarray
        Array to place the Cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing Cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out



def track_to_coords(xml_path,print_summary=True):
#    global time_unit
    tree =lineageTree(xml_path,file_type='TrackMate')
    t = ET.parse(xml_path)
    root = t.getroot()
    # get the pixel size
    units = root[1].attrib['spatialunits']
    assert units in ('micron','µm','um')
    img_settings = root[2][0].attrib
    um_per_px = np.array([img_settings['pixelwidth'],img_settings['pixelheight']],dtype=float)
    dt = float(root[2][0].attrib['timeinterval'])
    time_unit = root[1].attrib['timeunits']
    if time_unit == 'sec':
        dt/=60
    elif time_unit in ('h','hour','hours'):
        dt*=60
    tracks = tree.get_all_tracks()
    pos = [np.array([ tree.pos[x][:2] for x in tr ]) for tr in tracks  ] 
    time = [ np.array([ tree.time[x]*dt for x in tr ]) for tr in tracks]
    node_id = [ [x for x in tr] for tr in tracks]
    len_tracks = [len(tr) for tr in tracks]
    min_timepoints = 5 
    is_short_track = lambda x: x < min_timepoints
    num_small_tracks = sum(map(is_short_track,len_tracks))
    print(os.path.basename(xml_path))
    
    print(f'time step: {dt} min',f'pixel size: {um_per_px} um',
        f'#tracks: {len(tracks)}, ( {num_small_tracks} <  {min_timepoints} timepoints)',sep='\t')
    return (pos,time,node_id),tree
    

def get_speed(dat,by_id=False): 
    if by_id:  
        speeds = {}
    else:
        speeds = []
    for pos,time,node_ids in zip(*dat):
        #sd = np.sqrt((pos[:,0] - pos[0,0])**2 +(pos[:,1]-pos[0,1])**2)
        dr_vec = np.diff(pos,axis=0)
        dr = np.sqrt(dr_vec[:,0]**2 + dr_vec[:,1]**2)
        speed = dr/np.diff(time)
        if by_id:
            speed = np.concatenate(([0],speed)) # make equal length
            assert len(node_ids) == len(speed)
            for node_id,speed_node in zip(node_ids,speed):
                speeds[node_id] = speed_node 
        else:
            speeds.append(speed)
    return speeds

def get_avg_speed(dat): 
    speeds = []
    for pos,time,node_ids in zip(*dat):
        #sd = np.sqrt((pos[:,0] - pos[0,0])**2 +(pos[:,1]-pos[0,1])**2)
        dr_vec = np.diff(pos,axis=0)
        dr = np.sqrt(dr_vec[:,0]**2 + dr_vec[:,1]**2)
        speed = dr/np.diff(time)
        avg_speed = np.average(speed,axis=0)
        speeds.append(avg_speed)
    return np.array(speeds)

def get_max_displacement(dat): 
    max_distances = []
    for pos,time,node_ids in zip(*dat):
        x = pos[:,0]
        y = pos[:,1]
        xij = cartesian((x,x)) #contains all combinations (n*2) of r_i,r_j
        yij = cartesian((y,y)) #contains all combinations (n*2) of r_i,r_j
        dx_ij = xij[:,0] - xij[:,1]
        dy_ij = yij[:,0] - yij[:,1]
        distance = np.sqrt(dx_ij**2 + dy_ij**2)  
        max_distance = np.max(distance)
        max_distances.append(max_distance)
    return np.array(max_distances)

# def get_speed_by_id(dat): 
#     speeds = [] 
#     for pos,time,node_id in zip(*dat):
#         dr_vec = np.diff(pos,axis=0)
#         dr = np.sqrt(dr_vec[:,0]**2 + dr_vec[:,1]**2)
#         speeds.append(dr/np.diff(time))
#     return speeds
    
def get_displacement(dat):
    disps = []
    for pos,time,node_id in zip(*dat):
        disp = np.sqrt((pos[:,0] - pos[0,0])**2 +(pos[:,1]-pos[0,1])**2)
        disps.append(disp)
    return disps


def create_tree_ete3(tree,node_ids,node,it=0): 
    from ete3 import Tree
    ''' Makes a tree using ete3 standard methods'''
    if it == 0:
        # create the whole tree
        ete_tree = Tree(name='root')
        if node_ids == None:  
            node_ids = [nid for nid,t in tree.time.items() if t == 0 ]
            # for node_id in node_ids:
            #     node = ete_tree.add_child(name=node_id)
        create_tree_ete3(tree,node_ids,ete_tree,it=it+1)
        return ete_tree
    # if it > 5:
    #     return     
    else:  
        for node_id in node_ids:
            # create our own node
            if node.name != 'root':
                dist = tree.time[node_id] - tree.time[node.name]
                #dist = np.random.randint(1,10)
                #dist = 1
            else:
                dist = float(tree.time[node_id])
                dist = 1
            my_node = node.add_child(name=node_id,dist=dist)
            if tree.successor.get(node_id): 
                succs = tree.successor[node_id]
                # create ete3 child nodes 
                create_tree_ete3(tree,succs,my_node,it=it+1)
                
def node_to_tracks(tree):
    node_track = {}
    for i,tr in enumerate(tree.get_all_tracks()):
        for node in tr:
            node_track[node] = i
    return node_track



# Extracting fluorescence data  

def load_from_pickle(fname,list_of_ids,out=None):
    import pickle
    if out is None:  
        out = {}
    with open(fname,'rb') as f: 
        while True:
            try:
                dat = pickle.load(f)
                if (list_of_ids is None) or dat['id'] in list_of_ids:
                    out[dat['id']] = dat
            except Exception as e:
                print(e)
                break
    return out

def get_fluo_by_node(dat,filter_small=False):
    # construct list of points: 
    fluorescence = [] 
    fluorescence_by_node = {}
    all_nodes_ids = dat.keys()
    for nid in all_nodes_ids:
        #t = tree.time[nid]
        integ_brn = dat[nid]['crop']

        if filter_small:
            # if area of the cell is too small it's likely an mis-segmented cell 
            area = np.sum(~integ_brn.mask) 
            if area < 50*50:
                continue
        #avg_bg = dat[nid]['background'].mean()
        # Fit the background 
        fitted_bg = dat[nid]['fit_background']
        dat[nid]['adjusted_fluo'] = (dat[nid]['crop'])/fitted_bg 
        norm_brn = dat[nid]['adjusted_fluo'].sum()    
        fluorescence_by_node[nid] = norm_brn
    #fluorescence = np.array(fluorescence)
    return fluorescence_by_node

def get_avg_fluo_track(dat,fluo_by_node):
    fluo_avg = []
    for time,pos,node_ids in zip(*dat):
        fluo_track = [fluo_by_node[node_id] for node_id in node_ids]
        fluo_avg.append(np.nanmean(fluo_track))
    return np.array(fluo_avg)




def get_div_times(dat):
    div_times = []
    alltimes = np.concatenate(dat[1])
    min_time,max_time = np.min(alltimes),np.max(alltimes)
    for pos,time,node_id in zip(*dat):
        div_times.append(time[-1]-time[0])
    return np.array(div_times)

def get_valid_tracks(dat):
    valid = []
    alltimes = np.concatenate(dat[1])
    min_time,max_time = np.min(alltimes),np.max(alltimes)
    for pos,time,node_id in zip(*dat):
        if time[0] == min_time or time[-1] == max_time: 
            valid.append(False) 
        else: 
            valid.append(True)
    return np.array(valid)

def get_div_times(dat):
    div_times = []
    alltimes = np.concatenate(dat[1])
    min_time,max_time = np.min(alltimes),np.max(alltimes)
    for pos,time,node_id in zip(*dat):
        if time[0] == min_time or time[-1] == max_time: 
            div_times.append(-1) 
        else: 
            div_times.append(time[-1]-time[0])
    return np.array(div_times)

def plot_tree(tree):
    from ete3 import TreeStyle,NodeStyle
    ts = TreeStyle()
    ts.scale =  10
    ete_tree = create_tree_ete3(tree,None,None)
    ete_tree.render('%%inline',tree_style=ts,h=1000)


def render_pdf_notebook(path):
    import fitz
    from PIL import Image
    doc = fitz.open(path)
    MAX_PAGES = 1
    
    zoom = 1 # to increase the resolution
    mat = fitz.Matrix(zoom, zoom)
    
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img 
        # display images
        plt.figure(figsize=(7,7), facecolor="w")
        plt.xticks(color="white")
        plt.yticks(color="white")
        plt.tick_params(bottom = False)
        plt.tick_params(left = False)
    
        plt.imshow(img)
    
        if i > MAX_PAGES - 1:
            break

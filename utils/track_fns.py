import os.path
import numpy as np
from matplotlib import pyplot as plt
from LineageTree import lineageTree
import scipy.stats as stats
import xml.etree.ElementTree as ET
import os
from itertools import chain
from ete3 import TreeStyle,NodeStyle


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


def load_tree(xml_path):
    tree =lineageTree(xml_path,file_type='TrackMate')
    root = ET.parse(xml_path).getroot()
    units = root[1].attrib['spatialunits']
    assert units in ('micron','µm','um')
    img_settings = root[2][0].attrib
    um_per_px = np.array([img_settings['pixelwidth'],img_settings['pixelheight']],dtype=float)
    dt = float(root[2][0].attrib['timeinterval'])
    time_unit = root[1].attrib['timeunits']
    print(f'time step: {dt} {time_unit}',f'pixel size: {um_per_px} um')
    return tree,root

def track_to_coords(xml_path,print_summary=True):
#    global time_unit
    print(os.path.basename(xml_path))
    tree,root = load_tree(xml_path)
    # get the pixel size
    units = root[1].attrib['spatialunits']
    assert units in ('micron','µm','um')
    img_settings = root[2][0].attrib
    # um_per_px = np.array([img_settings['pixelwidth'],img_settings['pixelheight']],dtype=float)
    dt = float(root[2][0].attrib['timeinterval'])
    time_unit = root[1].attrib['timeunits']
    if time_unit == 'sec':
        dt/=60
    elif time_unit in ('h','hour','hours'):
        dt*=60
    tracks = tree.get_all_tracks()
    # remove spurious tracks
    #tracks = [ tr  for tr in tracks if len(tr) > 1]
    pos = [np.array([ tree.pos[x][:2] for x in tr ]) for tr in tracks  ] 
    time = [ np.array([ tree.time[x]*dt for x in tr ]) for tr in tracks]
    node_id = [ [x for x in tr] for tr in tracks]
    len_tracks = [len(tr) for tr in tracks]
    min_timepoints = 5 
    num_small_tracks = sum(map(lambda x: x < min_timepoints,len_tracks))
    
    print(f'#tracks: {len(tracks)}, ( {num_small_tracks} <  {min_timepoints} timepoints)',sep='\t')
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
        if fitted_bg is not None:
            dat[nid]['adjusted_fluo'] = (dat[nid]['crop'])/fitted_bg
        else:
            dat[nid]['adjusted_fluo'] = (dat[nid]['crop'])
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

def get_fluo_track(dat,fluo_by_node):
    fluo_tracks = []
    for time,pos,node_ids in zip(*dat):
        fluo_track = [fluo_by_node[node_id] for node_id in node_ids]
        fluo_tracks.append(fluo_track)
    return fluo_tracks

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
    type_divs = []
    alltimes = np.concatenate(dat[1])
    min_time,max_time = np.min(alltimes),np.max(alltimes)
    for pos,time,node_id in zip(*dat):
        # skip spurious tracks (only 1 timepoint)
        if len(pos) == 1:
            continue
        # remove tracks which start at the begining and split or are children
        # but haven't divided at the end

        if ((time[0] == min_time and time[-1] != max_time) or
               (time[-1] == max_time and time[0] != min_time)): 
            type_div = -1
        # remove lack of division
        elif time[-1] == max_time and time[0] == min_time:
            type_div = 0
        # clean division
        else:
            type_div = 1
        div_time = time[-1]-time[0]
        if div_time == (max_time - min_time) and type_div == 1:
            raise Exception('foo')
        
        type_divs.append(type_div)
        div_times.append(div_time)
    return np.array(div_times),np.array(type_divs)

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





def construct_tree(tree,node_id):
    outree = [[node_id]]
    for t in tree.time_nodes.keys():
        new_ids = []
        for e in tree.time_nodes[t]:
            if e in outree[-1]:
                print(t)
                new_ids += tree.successor[node_id]
        if new_ids != []:
            outree.append(new_ids)
    return outree

def construct_tree_w_time(tree,node_id,time=None):
    ''' construct a list of nodes at each timepoint, 
        duplicating entries that are not changed ''' 
    times = tree.time
    if type(node_id) == int:
        # init 
        time = times[node_id]
        outree = construct_tree_w_time(tree,[[node_id]],time+1)
        tmax = max(tree.time_nodes.keys())
        all_times = list(range(time,tmax+10))
        return dict(zip(all_times,outree))
    
    new_ids = []
    outree = node_id
    for e in outree[-1]:
        if tree.successor.get(e):
            successor = tree.successor[e]
#            if duplicate:
            for s in successor:
                if times[s] > time:
                    new_ids.append(e)
                else:
                    new_ids.append(s)
                     
    if new_ids != []:
        childs = construct_tree_w_time(tree,[new_ids],time+1)
        outree += childs
    return outree

def get_all_successors_wtime(node):
    out = [] 
    if tree.successor.get(node):
        succs = tree.successor[node]
        out.append((tree.time[node],node))
        for succ in succs: 
            out += get_all_successors_wtime(succ)
        return out 
    else:
        return []


def crawl_tree(tree,node_id): 
    ret = []  
    if tree.successor.get(node_id): 
        succ = tree.successor[node_id]
        for s in succ:
            ret +=[crawl_tree(s)]
        return [node_id,ret]
    else:
        return [node_id]
    
    
def parent_child_from_node(tree,node_id): 
    ret = tuple()
    # if node_id == None:
    #     nodes = [nid,t for nid,t in tree.time if t == 0 ]
    #     ret += parent_child_from_node(tree,nid)
    #     return ret 
    if tree.successor.get(node_id): 
        succ = tree.successor[node_id]
        for s in succ:
            ret += parent_child_from_node(tree,s)
        ret += tuple((node_id,s,tree.time[s]-tree.time[node_id]) for s in succ)
    return ret


def crawl_tree_newick(tree,node_id,it=0): 
    ''' Makes a newick tree spanning our original node''' 
    if tree.successor.get(node_id): 
        ret = ()
        succ = tree.successor[node_id]
        for s in succ:
            obj = crawl_tree_newick(tree,s,it=it+1)
            if obj: 
                ret += (obj,)
        if len(ret) == 1:
            toreturn = str( "("+ret[0]+")" + str(node_id) )
        elif len(ret) == 0:
            return str(node_id)            
        else:
            toreturn = ""
            for r in ret:
                toreturn += "("+r+"),"
            toreturn = toreturn.removesuffix(",")
            toreturn += str(node_id)+""
        if it == 0:
            toreturn = "("+ toreturn + ");" 
        return toreturn 
    else:
        return str(node_id)

    
                
def create_tree_ete4(tree,node_ids,node,it=0): 
    from ete4 import Tree
    ''' Makes a tree using ete3 standard methods'''
    if it == 0:
        # create the whole tree
        ete_tree = Tree({'name': 'root'})
        if node_ids == None:  
            node_ids = [nid for nid,t in tree.time.items() if t == 0 ][:5]
            # for node_id in node_ids:
            #     node = ete_tree.add_child(name=node_id)
        create_tree_ete4(tree,node_ids,ete_tree,it=it+1)
        return ete_tree

    else:  
        for node_id in node_ids:
            # create our own node
            if node.name != 'root':
                dist = tree.time[node_id] - tree.time[int(node.name)]
            else:
                dist = tree.time[node_id]
            my_node = node.add_child(name=node_id,dist=dist)
            if tree.successor.get(node_id): 
                succs = tree.successor[node_id]
                # create ete3 child nodes 
                create_tree_ete4(tree,succs,my_node,it=it+1)        
                    

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



    

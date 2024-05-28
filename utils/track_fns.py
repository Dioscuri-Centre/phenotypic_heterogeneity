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

def track_to_coords(xml_path,remove_spurious=True,print_summary=True):
    debug = True
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
    # remove spurious tracks (this includes removing the nodes etc, there's a bultin method for that
    if remove_spurious:
        suspicious_tracks = [ tr  for tr in tree.get_all_tracks() if len(tr) == 1]
        spurious = []
        for tr in suspicious_tracks:
            nid = tr[0]
            if debug:
                print(f'suspicious node {nid} time {tree.time[nid]}')
            
            # if not (nid in tree.successor or nid in tree.predecessor):
            #     spurious.append(tree.remove_node(nid))
            #    if debug:
            #       print(f'removed {spurious[-1]} node {nid} time {tree.time[nid]}')
            tree.remove_track(tr)
            #spurious.append(tree.remove_node(nid)) # delete len-1 tracks even if they are real ones
            
        print(f'removed {len(spurious)} spurious tracks out of  {len(suspicious_tracks)} suspicious ones ')
         #need to manually delete,there's a bug in the library currently
        del tree._all_tracks 
    tracks = tree.get_all_tracks()
    node_id = [ [x for x in tr] for tr in tracks]
    pos = [np.array([ tree.pos[x][:2] for x in tr ]) for tr in tracks  ]
    time = [ np.array([ tree.time[x]*dt for x in tr ]) for tr in tracks]
    
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
        avg_speed = np.nanmean(speed,axis=0)
        speeds.append(avg_speed)
    return np.array(speeds)

def get_max_speed(dat):
    speeds = []
    for pos,time,node_ids in zip(*dat):
        #sd = np.sqrt((pos[:,0] - pos[0,0])**2 +(pos[:,1]-pos[0,1])**2)
        dr_vec = np.diff(pos,axis=0)
        dr = np.sqrt(dr_vec[:,0]**2 + dr_vec[:,1]**2)
        speed = dr/np.diff(time)
        max_speed = np.max(speed,axis=0)
        speeds.append(max_speed)
    return np.array(speeds)

def get_speed(dat):
    speeds = []
    for pos,time,node_ids in zip(*dat):
        #sd = np.sqrt((pos[:,0] - pos[0,0])**2 +(pos[:,1]-pos[0,1])**2)
        dr_vec = np.diff(pos,axis=0)
        dr = np.sqrt(dr_vec[:,0]**2 + dr_vec[:,1]**2)
        speed = dr/np.diff(time)
        #avg_speed = np.average(speed,axis=0)
        speeds.append(speed)
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
                #dist = 1
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

def get_fluo_by_node(dat,filter_small=False,normed=False):
    # construct list of points:
    fluorescence = []
    fluorescence_by_node = {}
    all_nodes_ids = dat.keys()
    for nid in all_nodes_ids:
        #t = tree.time[nid]
        integ_brn = dat[nid]['crop']
        area = np.sum(~integ_brn.mask)
        if filter_small:
            # if area of the cell is too small it's likely an mis-segmented cell
            
            if area < 50*50:
                #continue
                fluorescence_by_node[nid] = np.nan
                continue
        #avg_bg = dat[nid]['background'].mean()
        # Fit the background
        fitted_bg = dat[nid]['fit_background']
        if fitted_bg is not None:
            assert fitted_bg.max() != 0
            dat[nid]['adjusted_fluo'] = (dat[nid]['crop'])/fitted_bg
        else:
            dat[nid]['adjusted_fluo'] = (dat[nid]['crop'])
        norm_brn = dat[nid]['adjusted_fluo'].sum()
        fluorescence_by_node[nid] = norm_brn
    #fluorescence = np.array(fluorescence)
    return fluorescence_by_node

def get_avg_fluo_track(dat,fluo_by_node):
    fluo_avg = []
    for fluo in get_fluo_track(dat,fluo_by_node):
        fluo_avg.append(np.nanmean(fluo))
    return np.array(fluo_avg)

def get_max_fluo_track(dat,fluo_by_node):
    fluo_max = []
    for fluo in get_fluo_track(dat,fluo_by_node):
        fluo_max.append(np.nanmax(fluo))
    return np.array(fluo_max)

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


def unzip(x):
    return list(zip(*x))

def get_nonspurious_tracks(dat):
    return list(zip(*[ e for e in zip(*dat) if len(e[0]) > 1]))
def get_div_times_lintree(dat,remove_spurious=True):
    div_times = []
    type_divs = []
    alltimes = np.concatenate(dat[1])
    min_time,max_time = np.min(alltimes),np.max(alltimes)
    for pos,time,node_id in zip(*dat):
        # skip spurious tracks (only 1 timepoint)
        if remove_spurious and len(pos) == 1:
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




def pool_dat(dats,fluos=None):
    """
    pools multiple data, eventually with fluorescence data  
    .. Keyword Arguments:
    :param dats: the data (contains tracks coordinates, timepoints etc)
    :param fluos: optional fluorescence data (default: None) 

    .. Types:
    :type dats: list
    :type fluos: list

    .. Returns:
    :return:
    :rtype:

    """
    from itertools import chain
    debug = False
    pos_pooled,times_pooled,node_ids_pooled = (e.copy() for e in dats[0])
    fluos_pooled = {}
    if fluos is not None:
        assert len(dats) == len(fluos)
        fluos_pooled = fluos[0].copy()

    for i,(pos,times,node_ids) in enumerate(dats[1:],start=1):
        pos_pooled.extend(pos)
        times_pooled.extend(times)
        if debug: print(len(pos_pooled))
        set_old_node_ids = set(chain(*node_ids))
        set_node_ids_pooled = set(chain(*node_ids_pooled))

        if set_old_node_ids.intersection(set_node_ids_pooled) == set():
            node_ids_pooled.extend(node_ids)
            fluos_pooled.update(fluos[i])
        else:
            # we need to change the node_ids, simplest way is to offset_it
            offset = np.max(list(set_node_ids_pooled))
            if debug:
                print(f'applied offset {offset}')
                print(f'common node ids: {set_old_node_ids.intersection(set_node_ids_pooled)}')
            set_new_node_ids = offset+ np.array(list(set_old_node_ids))
            new_node_ids = [ [v+offset for v in e] for e in node_ids]
            node_ids_pooled.extend(new_node_ids)
            if fluos is not None:
                fluo = fluos[i]
                new_fluo = { k:fluo[kold] for k,kold in zip(set_new_node_ids,set_old_node_ids) }
                fluos_pooled.update(new_fluo)
    assert len(pos_pooled) == sum(len(x[0]) for x in dats)
    assert len(times_pooled) == sum(len(x[1]) for x in dats)
    ret =  (pos_pooled,times_pooled,node_ids_pooled),fluos_pooled
    return ret




# code found in https://stackoverflow.com/a/39782685
def arrowplot(axes, x, y, nArrs=30, mutateSize=10, color='gray', markerStyle='o'): 
    '''arrowplot : plots arrows along a path on a set of axes
        axes   :  the axes the path will be plotted on
        x      :  list of x coordinates of points defining path
        y      :  list of y coordinates of points defining path
        nArrs  :  Number of arrows that will be drawn along the path
        mutateSize :  Size parameter for arrows
        color  :  color of the edge and face of the arrow head
        markerStyle : Symbol
    
        Bugs: If a path is straight vertical, the matplotlab FanceArrowPatch bombs out.
          My kludge is to test for a vertical path, and perturb the second x value
          by 0.1 pixel. The original x & y arrays are not changed
    
        MHuster 2016, based on code by 
    '''
    import matplotlib.patches as patches

    # recast the data into numpy arrays
    x = np.array(x, dtype='f')
    y = np.array(y, dtype='f')
    nPts = len(x)

    # Plot the points first to set up the display coordinates
    axes.plot(x,y, markerStyle, ms=5, color=color)

    # get inverse coord transform
    inv = axes.transData.inverted()

    # transform x & y into display coordinates
    # Variable with a 'D' at the end are in display coordinates
    xyDisp = np.array(axes.transData.transform(list(zip(x,y))))
    xD = xyDisp[:,0]
    yD = xyDisp[:,1]

    # drD is the distance spanned between pairs of points
    # in display coordinates
    dxD = xD[1:] - xD[:-1]
    dyD = yD[1:] - yD[:-1]
    drD = np.sqrt(dxD**2 + dyD**2)

    # Compensating for matplotlib bug
    dxD[np.where(dxD==0.0)] = 0.1


    # rtotS is the total path length
    rtotD = np.sum(drD)

    # based on nArrs, set the nominal arrow spacing
    arrSpaceD = rtotD / nArrs

    # Loop over the path segments
    iSeg = 0
    while iSeg < nPts - 1:
        # Figure out how many arrows in this segment.
        # Plot at least one.
        nArrSeg = max(1, int(drD[iSeg] / arrSpaceD + 0.5))
        xArr = (dxD[iSeg]) / nArrSeg # x size of each arrow
        segSlope = dyD[iSeg] / dxD[iSeg]
        # Get display coordinates of first arrow in segment
        xBeg = xD[iSeg]
        xEnd = xBeg + xArr
        yBeg = yD[iSeg]
        yEnd = yBeg + segSlope * xArr
        # Now loop over the arrows in this segment
        for iArr in range(nArrSeg):
            # Transform the oints back to data coordinates
            xyData = inv.transform(((xBeg, yBeg),(xEnd,yEnd)))
            # Use a patch to draw the arrow
            # I draw the arrows with an alpha of 0.5
            p = patches.FancyArrowPatch( 
                xyData[0], xyData[1], 
                arrowstyle='simple',
                mutation_scale=mutateSize,
                color=color, alpha=0.5)
            axes.add_patch(p)
            # Increment to the next arrow
            xBeg = xEnd
            xEnd += xArr
            yBeg = yEnd
            yEnd += segSlope * xArr
        # Increment segment number
        iSeg += 1


def bin_xy_data(data,qx=0.33,dropnan=False,bincounts=False):
    from numpy.ma import MaskedArray
    # data in the form of a list of pairs of lists [x] and [y] of coordinates
    xx,yy=[[]],[[]]
    for n,d in enumerate(data):
        x,y=d
        for i in range(0,len(x)):
            if dropnan is True:                
                if (np.isnan(x[i]) or np.isnan(y[i])
                    or isinstance(x[i],MaskedArray)
                    or isinstance(y[i],MaskedArray)) :
                    continue
            j=int(x[i]/qx)
            if (j<len(xx)):
                yy[j].append(y[i])
                xx[j].append(x[i])
            else:
                yy.append([])
                xx.append([])
    # if dropnan:
    #     return [np.nanmean(x) for x in xx],[np.nanmean(y) for y in yy], [np.std(y)/np.sqrt(sum(~np.isnan(y))) for y in yy]
    if bincounts:
        return [np.mean(x) for x in xx],[np.mean(y) for y in yy],[np.std(y)/np.sqrt(len(y)) for y in yy],[ len(x) for x in xx]
    return [np.mean(x) for x in xx],[np.mean(y) for y in yy],[np.std(y)/np.sqrt(len(y)) for y in yy]
    


def get_div_times(tree):
    nodes = tree.get_leaves()
    div_times,div_nodes = [],[]
    while(len(nodes) > 0):
        n = nodes.pop()
        #print(n.name)
        for n2 in n.get_ancestors():
            if n2.get_sisters():
                ancestor = n.get_common_ancestor(n2.get_sisters())
                dist = ancestor.get_distance(tree.get_tree_root())
                if (ancestor not in div_nodes) and (dist > 1.0):
                    div_times.append(dist-1.0)
                    div_nodes.append(ancestor)
                    nodes.append(ancestor)
                break
    return div_times,div_nodes


def get_death_times(tree):
    death_times = [] 
    death_nodes = [] 
    max_dist = 1.0
    for n in tree.traverse('postorder'):
        max_dist = max(max_dist,n.get_distance(tree.get_tree_root()))
    
    for n in tree.get_leaves():
        dist =  n.get_distance(tree.get_tree_root())
        if dist != max_dist:
            #print(dist)
            death_times.append(dist-1.0)
            death_nodes.append(n)
    return death_times,death_nodes


def get_pop_times(tree):
    ''' returns the population times for each nodes, to be used by numpy unique'''
    pop_times = [] 
    
    for n in tree.traverse('postorder'):
        dist =  n.get_distance(tree.get_tree_root())
        pop_times.append(dist-1.0)
    return pop_times


from detectron2.structures import PolygonMasks,BitMasks
from detectron2.structures import polygons_to_bitmask
from detectron2.structures.boxes import Boxes
from detectron2.utils.visualizer import GenericMask
from detectron2.structures.instances import Instances
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import pickle
import cv2
import numpy as np
from mltools.utils_images import bluecmap,redcmap
import os.path
import dask.array

from mltools.segment_anything_fns import (detect_crops_SAM,save_to_ImageJ_ROIs,
                                          predict_image_to_det,
                                          translate_polygonmask)
from segment_anything import sam_model_registry, SamPredictor
from mltools.batch_processing import open_dets
from itertools import chain
import dask
import pickle

def annos_to_instances(elt,dims):
    ''' convert detectron2 annotations to instances ''' 
    masks = [GenericMask(x['segmentation'],*dims).mask for x in elt['annotations']]
    boxes = [x['bbox'] for x in elt['annotations']]
    boxes = torch.tensor(boxes)
    boxes = Boxes(boxes)
    annos = {'pred_masks':np.array(masks),'pred_boxes':boxes}
    return Instances(dims,**annos)

def correct_bg(img,bg,insts):
    ''' Takes detected objects (a.k.a instances) and correct the background within those detections
        also, remove the background '''
    w = 300 
    pred_masks = insts.pred_masks
    if type(pred_masks) == torch.tensor: 
        cells_mask = pred_masks.sum(axis=0).cpu().numpy() > 0
    else: 
        cells_mask = pred_masks.sum(axis=0) > 0
    cells = (img*cells_mask).astype(float)
    return cells/bg

def plot_corrected_cells(img,bg,insts): 
    w = 300 
    N = len(insts)
    fig,ax = plt.subplots(1,N,figsize=(5*N,5),layout='constrained')
    fig2,ax2 = plt.subplots(1,N,figsize=(5*N,5),layout='constrained')
    fig.suptitle('cell brightness normalized by bg value')
    fig2.suptitle('fitted background')
    
    cells_corrected = correct_bg(img,bg,insts)
    for i in range(N): 
        # Take the center of the cell
        xc,yc = insts.pred_boxes.get_centers()[i].tolist()
        xc,yc = int(xc),int(yc)
        # normalize brightness 
        cell_reg = cells_corrected[yc-w//2:yc+w//2,xc-w//2:xc+w//2,]
        bg_reg = bg[yc-w//2:yc+w//2,xc-w//2:xc+w//2,]
        ax[i].imshow(np.ma.masked_values(cell_reg,0),cmap=redcmap)
        ax[i].set_facecolor('black')
        ax2[i].imshow(bg_reg,cmap=redcmap)
        ax2[i].set_facecolor('black')


def trackmate_track_to_detections(tree,root,imgs,out_bin,s=500):
    ''' converts a trackmate track to a detection object
        Arguments:
            - the lineage tree
            - the images''' 

    units = root[1].attrib['spatialunits']
    assert units in ('micron','µm','um')
    img_settings = root[2][0].attrib
    um_per_px = np.array([img_settings['pixelwidth'],img_settings['pixelheight']],dtype=float)
    sam_checkpoint = "/home/idjafc/Code/segment-anything/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    if os.path.isfile(out_bin):
        return open_dets(out_bin)
    
    dets = [ ]
    
    with open(out_bin,'wb') as f: 
        for t in tqdm(list(tree.time_nodes.keys())):
            nodes_id = list(tree.time_nodes[t])
            if not nodes_id: continue # there might be 0 detection 
            nodes_pos = np.array([tree.pos[idx][:2] for idx in nodes_id])
            nodes_pos /= um_per_px
            if type(imgs) == dask.array.Array:
                img = imgs[t].compute()
            else: 
                img = imgs[t]
            h,w = img.shape
            # Let's go through all objects by cropping a window around them 
            dets_t = []
            for p in nodes_pos:
                # Be careful with the order of things
                xs,xe,ys,ye = (max(p[0]-s//2,0),min(w,p[0]+s//2),
                               max(p[1]-s//2,0),min(h,p[1]+s//2),
                              )
                ys,ye,xs,xe = int(ys),int(ye),int(xs),int(xe)
                reg = img[ys:ye,xs:xe]
                p = p - np.array([xs,ys])
                point_rois = torch.tensor(p[None,None,:],device='cuda')
                det = predict_image_to_det(predictor,reg,point_rois=point_rois)
                translate_polygonmask(det.polys,xs,ys)
                boxes = det.instances_all.pred_boxes
                boxes[:,0] += xs
                boxes[:,1] += ys
                boxes[:,2] += xs
                boxes[:,3] += ys
                det.instances_all._image_size = img.shape
                dets_t.append(det)
            dets_t[0].instances_all = Instances.cat([d.instances_all for d in dets_t])
            dets_t[0]._polys = list(chain.from_iterable([d._polys for d in dets_t]))
            dets_t[0].coords = {'c':0,'z':0,'t':t}
            dets_t[0].instances_all._image_size = img.shape
            dets_t[0].instances_all.node_id = nodes_id
            dets.append(dets_t[0])
            pickle.dump(dets[-1].save(None),f)
    return dets



def extract_reg(insts,w,arr):
    ''' Extracts region from an array of detected objects around their center 
        (determined by their bounding box) '''
    regs = []
    if type(insts.pred_boxes) == Boxes: 
        boxes = insts.pred_boxes
    else: 
        boxes = Boxes(insts.pred_boxes) 
    for i in range(len(insts)): 
        # Take the center of the cell
        xc,yc = boxes.get_centers()[i].tolist()#.cpu().numpy():
        xc,yc = int(xc),int(yc)
        regs.append(arr[yc-w//2:yc+w//2,xc-w//2:xc+w//2,])
    
    # We next pad the array
    target_shape = np.array((w,w)) 
    for i,reg in enumerate(regs):
        #to_pad = np.maximum(target_shape-np.array(reg.shape),0)
        # right pad the array (doesn't really matter for now where to pad)
        #regs[i] = np.pad(reg,[(0,d) for d in to_pad])
        regs[i] = pad_if_necessary(regs[i],target_shape)
    return regs

def pad_if_necessary(arr,target_shape): 
    to_pad = np.maximum(target_shape-np.array(arr.shape),0)
    # right pad the array (doesn't really matter for now where to pad)
    return np.pad(arr,[(0,d) for d in to_pad])
    

def extract_reg_from_det(det,img,win_size=500):
    insts = det.instances_all 
    height,width = insts.image_size 
    polygonmasks = PolygonMasks(det.polys)
    RES = BitMasks.from_polygon_masks(polygonmasks,height=height,width=width)
    glob_mask = RES.tensor.numpy().sum(axis=0)
    rets = extract_reg(insts,win_size,img*glob_mask)
    return rets

def translate_polygonmask_cpy(polygons,x,y):
    ret = [] 
    for obj in polygons: 
        ret.append([])
        for p in obj:
            # each obj is a list of polygons (might contain holes)
            obj_tr = p.copy()
            obj_tr[0::2]+=x
            obj_tr[1::2]+=y
        ret[-1].append(obj_tr)
    return ret 

def extract_reg_from_det_single_masks(det,img,img_bf,flatfield,win_size=500):
    insts = det.instances_all
    height,width = insts.image_size 
    polygonmasks = PolygonMasks(det.polys)
    rets = []
    for i,poly in enumerate(polygonmasks):
        poly_to_crop = [poly]
        xc,yc = Boxes(insts[i].pred_boxes).get_centers().squeeze().to(int).tolist() 
        poly_to_crop = translate_polygonmask_cpy(poly_to_crop,-xc+win_size//2,-yc+win_size//2)
        bmask = BitMasks.from_polygon_masks(poly_to_crop,height=win_size,width=win_size)
        mask_reg = bmask.tensor.numpy().squeeze()
        # # crop the mask and the image 
        cell_crop = extract_reg(insts[i],win_size,img)[0]
        cell_bf_crop = extract_reg(insts[i],win_size,img_bf)[0]
        # Then create masked values
        cell_reg = np.ma.masked_array(cell_crop,~mask_reg) #mask it 
        cell_bf_reg = np.ma.masked_array(cell_bf_crop,~mask_reg) #mask it 
        bg_reg = np.ma.masked_array(cell_crop,mask=mask_reg)
        #with warnings.catch_warnings():
            #warnings.simplefilter("ignore")
        if flatfield:
            fit_bg_reg = fit_poly_ndeg_try2(bg_reg,n=4,mask=cell_reg.mask)[0]
        else:
            fit_bg_reg = None
        yield cell_reg,bg_reg,fit_bg_reg,cell_bf_reg,poly_to_crop
        
def load_from_pickle(fname,list_of_ids,out=None):
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



def old_process_det(fnames_template,lock,det,imgs=None,imgs_bf=None,crop=None,f=None):
    t = det.coords['t']
    if imgs is not None and fnames_template is not None :
        raise Exception("cannot provide both template and imgs")
    if imgs is not None:
        img = imgs[t]
    else: 
        img = imread(fnames_template.format(t))

    if imgs_bf is not None:
        img_bf = imgs_bf[t]
    if crop is not None:
        assert type(crop) == slice
        img = img[crop]
        img_bf = img_bf[crop]

    if type(img) == dask.array.Array:
        img = img.compute()
    if type(img_bf) == dask.array.Array:
        img_bf = img_bf.compute()
        
    #crop = img #[5687:5687+10920,3815:3815+11088] # for well 7 [3749:3749+11136,2784:2784+13088]
    #crop_bf = img_bf
    regs = list(extract_reg_from_det_single_masks(det,img,img_bf))
    node_ids = det.instances_all.node_id          
    # with lock:
    #     for node_id,(cell,bg,fit_bg,crop_bf,poly) in zip(node_ids,regs):
    #         pickle.dump({'id':node_id,'crop':cell,'background':bg,'fit_background':fit_bg,'crop_bf':crop_bf,'poly':poly},f)




def process_det(det,img,img_bf,f,lock,fname=None,crop=None,flatfield=True):

    if img is None:
        img = imread(fnames_template)

    if crop is not None:
        assert type(crop) == slice
        img = img[crop]
        img_bf = img_bf[crop]
    if type(img) == dask.array.Array:
        img = img.compute()
    if type(img_bf) == dask.array.Array:
        img_bf = img_bf.compute()

    #crop = img #[5687:5687+10920,3815:3815+11088] # for well 7 [3749:3749+11136,2784:2784+13088]
    regs = list(extract_reg_from_det_single_masks(det,img,img_bf,flatfield))
    node_ids = det.instances_all.node_id          
    with lock:
        for node_id,(cell,bg,fit_bg,crop_bf,poly) in zip(node_ids,regs):
            pickle.dump({'id':node_id,'crop':cell,'background':bg,'fit_background':fit_bg,'crop_bf':crop_bf,'poly':poly},f)


import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import curve_fit
from functools import partial 


def make_grid(m):
    ny,nx = m.shape
    xs,ys = np.arange(nx),np.arange(ny)
    x,y = np.meshgrid(xs,ys)
    return x,y


def poly_ndeg_try2(n,x,*args):
    ret = 0 
    for i in range(n+1):
        for j in range(n+1):
            ret += args[i*(n+1)+j]*pow(x[0],i)*pow(x[1],j) #+ args[i+2]*pow(args[i+3]-x[1],i)
    return ret

def fit_poly_ndeg_try2(bg,n,mask=None):
    if (mask is None):
        mask = np.ones_like(bg,dtype=bool)
    xs,ys = make_grid(bg)
    fn = partial(poly_ndeg_try2,n)
    init_guess = np.array([1]*(n+1)**2)
    params = curve_fit(fn,np.stack((xs[mask],ys[mask])),bg[mask],p0=init_guess)[0]
    fitted = fn((xs,ys),*params)
    return fitted,xs,ys,params

def poly_ndeg(n,x,*args):
    ret = 0 
    for i in range(n+1):
        ret += args[i]*pow(args[i+1]-x[0],i) + args[i+2]*pow(args[i+3]-x[1],i)
    return ret 
def fit_poly_ndeg(bg,n,mask=None):
    if (mask is None):
        mask = np.ones_like(bg,dtype=bool)
    xs,ys = make_grid(bg)
    fn = partial(poly_ndeg,n)
    init_guess = np.array([1]*4*(n+1))
    params = curve_fit(fn,np.stack((xs[mask],ys[mask])),bg[mask],p0=init_guess)[0]
    fitted = fn((xs,ys),*params)
    return fitted,xs,ys,params




# We need at least a 3rd degree polynomial 


def fit_poly_ndeg_better(bg,n,mask=None):
    if (mask is None):
        mask = np.ones_like(bg,dtype=bool)
    X,Y = make_grid(bg)

    polys = [ X**i*Y**j for i in range(n+1) for j in range(n+1)]
    #polys.pop(0)
    polys_masked = [ p[mask] for p in polys]

    #A = np.array([1*X**0, X, Y, X**2, X**2*Y, X**2*Y**2, 
    #                            X**3,X**3*Y,X**3*Y**2,X**3*Y**3,
    #                            X**4,X**4*Y,X**4*Y**2,X**4*Y**3,X**4*Y**4,
    #                   Y**2, Y**2*X, Y**2*X**2, Y**3,Y**3*X,Y**3*X**2]).T
    A = np.array(polys_masked).T
    vals = bg[mask]
    coeff, r, rank, s = np.linalg.lstsq(A,vals,)
    fitted = (np.array(polys).T*coeff).sum(axis=-1)#.reshape(X.shape)
    #print(X.shape,np.array(polys).T.shape)
    return fitted,xs,ys,coeff


# Filter images by their node id
def filterid(vals,e):
    if e['id'] in vals:
        return True
    else:
        return False

def plot_all_imgs_time_t(t,treenode,data,out=None):
    nwin = max([len(x) for x in treenode.values()])
    imgs = [ data[node]['crop'] for node in treenode[t] if data.get(node) ]
    if not imgs:
        return 'err' 
    n = len(imgs)
    fig,axs = plt.subplots(1,nwin,squeeze=False)
    fig.suptitle(f't={t}')
    for i in range(n):
        axs[0,i].imshow(imgs[i],cmap=redcmap)
    if out:
        fig.savefig(out,dpi=200)
        plt.close(fig)
    else:
        return fig,axs

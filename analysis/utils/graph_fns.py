import networkx as nx
import numpy as np
import pandas as pd

def get_nbranches_tree(trees):
    def get_nbranches(tree,n):
        succs = list(tree.successors(n))
        while len(succs) == 1:
            n = succs[0]
            succs = list(tree.successors(n))
        if len(succs) == 0:
            return 1
        elif len(succs) > 1:
            return sum(get_nbranches(tree,n) for n in succs)+1
    G = combine_trees(trees)
    # find root nodes 
    root_nodes = [x for x in G.nodes() if G.in_degree(x)==0]
    trees_len = []
    for root_node in root_nodes:
        tree= G.subgraph(nx.descendants(G,root_node) | {root_node})
        assert nx.is_tree(tree)
        trees_len.append(get_nbranches(tree,root_node))
    return trees_len

def get_ndiv_tree(trees):
    G = combine_trees(trees)
    # find root nodes 
    root_nodes = [x for x in G.nodes() if G.in_degree(x)==0]
    trees_ndiv = []
    for root_node in root_nodes:
        ndiv = 0
        for n in nx.descendants(G,root_node):
            if G.out_degree(n) > 1:
                ndiv+=1
        trees_ndiv.append(ndiv)
    return trees_ndiv

def get_next_branch(n,g):    
    succs = list(g.successors(n))
    cnt = 0
    nodes = [n] 
    while len(succs) == 1:
        n = succs[0]
        nodes.append(n)
        succs = list(g.successors(n))
    return nodes,succs

def get_prev_branch(n,g):
    succs = list(g.successors(n))
    preds = list(g.predecessors(n))
    cnt = 0
    nodes = [n] 
    while len(succs) < 2:
        n = preds[0]
        preds = list(g.predecessors(n))
        succs = list(g.successors(n))
        nodes.append(n)
        if len(preds) == 0:
            break
    return nodes,preds

def get_path_to_root(n,g):
    preds = list(g.predecessors(n))
    nodes = [n] 
    while len(preds) >= 1:
        n = preds[0]
        preds = list(g.predecessors(n))
        nodes.append(n)
        if len(preds) == 0:
            break
    return nodes


def compute_coord_one_tree(tree,n,pad,coords,g):
    ''' takes a node an compute the x coordinate, if a split occured, add offset'''
    # Assume constant width for branches
    nodes,succs = get_next_branch(n,tree)
    n_next = nodes[-1]
    g.add_node(n_next,**tree.nodes[n_next])
    g.nodes[n_next]['xplot'] = g.nodes[n]['xplot']
    # vertical edge
    g.add_edge(n,n_next)
    # deadend
    if len(succs) == 0:
        coords.append(g.nodes[n]['xplot'])

    elif len(succs) > 1:
        nsuccs = len(succs)
        for i,n_succs in enumerate(succs):
            g.add_node(n_succs,**tree.nodes[n_succs])
            g.add_edge(n_next,n_succs)
            g.nodes[n_succs]['xplot'] = g.nodes[n_next]['xplot'] + (-0.5+i/(nsuccs-1))*pad
            compute_coord_one_tree(tree,n_succs,0.45*pad,coords,g)
        
def make_simple_graph(g,nmax,min_num_nodes,root_nodes=None):
    branch_pad = 1
    tree_pad = 1.0
    if root_nodes:
        pass
    else:
        root_nodes = [n for n in g.nodes if len(nx.descendants(g,n)) > min_num_nodes and g.in_degree(n) == 0]
    xnew = 0 
    newgraph = nx.DiGraph()
    for n in root_nodes[:nmax]:
        tree = get_tree_from_root(g,n)
        newgraph.add_node(n,**g.nodes[n])
        newgraph.nodes[n]['xplot'] = xnew
        coords = []
        compute_coord_one_tree(tree,n,branch_pad,coords,newgraph)
        xnew = max(coords) + tree_pad
    return newgraph


def plot_simple_graph(g,labels=True): 
    # construct the subgraph with only necessary nodes 
    pos = pd.DataFrame.from_records(list(g2.nodes.values()),index=list(g2.nodes))
    edges = np.array([ (pos.loc[n].xplot,pos.loc[n2].xplot,pos.loc[n].t,pos.loc[n2].t) for n,n2  in g2.edges])
    plt.plot(edges[:,:2].T,edges[:,2:].T,color='black')
    if labels:
        text = np.array([ (pos.loc[n].xplot,pos.loc[n].t,n) for n in g2.nodes if g2.out_degree(n) > 1])
        for x,y,lab in text:
            plt.text(x,y,s=lab.astype(str),fontdict=dict(size=10))

def combine_trees(trees):
    G = nx.DiGraph()
    for t in trees:
        assert nx.is_tree(t)
        G.update(t)
    return G

def read_graph_mastodon(fname):
    from mastodon_reader import MastodonReader
    mr = MastodonReader(fname)
    # show meta data
    meta_data = mr.read_metadata()
    # read (networkX) graph representation, spot and link tables with features and tags columns
    graph, spots, links, tag_definition = mr.read(tags=True, features=True)
    return graph,spots,links,tag_definition

def get_tagged_trees(graph,spots,tags=['untracked','verified'],asgraph=False):
    # the tagged nodes
    tagged_nodes = [ spot for tag in tags for spot in spots[spots[tag+'_NAME'] != ''].index.to_list() ] 
    trees = get_trees_containing_node(graph,tagged_nodes)
    if asgraph:
        subgraph = graph.edge_subgraph(e for tree in trees for e in tree.edges )
        assert nx.is_branching(subgraph)
        return subgraph
    return trees 


def get_all_trees(graph,asgraph=False):
    # the tagged nodes
    root_nodes = [ n for n in graph.nodes if graph.in_degree(n) == 0 ]
    trees = [ get_tree_from_root(graph,n) for n in root_nodes]
    if asgraph:
        subgraph = graph.edge_subgraph(e for tree in trees for e in tree.edges )
        #assert nx.is_branching(subgraph)
        return subgraph

    return trees 


def get_trees_containing_node(graph,tagged_nodes):
    trees = []
    root_nodes = []
    for n in tagged_nodes:
        # generate the subgraph spanned by the the root node
        root_node = [x for x in nx.ancestors(graph,n) if graph.in_degree(x) == 0  ]
        # it may be that the current node is the root one
        if root_node == []:
            root_node = [n]
        assert  len(root_node) == 1
        root_node = root_node[0]
        if root_node in root_nodes:
            # skip already generated trees
            continue
        else:
            root_nodes.append(root_node)
            
        # If we try to get the spanning tree, some edges are missing
        # tree = nx.minimum_spanning_tree(subg.to_undirected())
        # E = set(tree.edges())  # optimization
        # oriented_edges = [e for e in subg.edges() if e in E or reversed(e) in E]
        # subg = graph.subgraph(nodes)
        trees.append(get_tree_from_root(graph,root_node))
    return trees


def label_nodes_by_id(G):
    for n in G.nodes:
        for k in list(G.nodes[n].keys()):
            G.nodes[n].pop(k,None)
        G.nodes[n]['ID'] = n

def plot_trees_pydot(trees):
    from networkx.drawing.nx_pydot import graphviz_layout
    subgraph = combine_trees(trees)
    label_nodes_by_id(subgraph)
    pos = nx.nx_pydot.graphviz_layout(subgraph,prog='dot')
    nx.draw(subgraph,node_size=5, arrowstyle='-',
            pos=pos
           )

def remove_cycles(graph):
    # remove all trivial cycles
    # cf: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html#networkx.algorithms.cycles.simple_cycles
    
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes_to_delete = set()
    for n in graph.nodes():
        predecessors = list(graph.predecessors(n))
        if len(predecessors) > 1: 
            # take one the predecessor with the longest list of successors 
            # branches = [get_prev_branch(n,graph)[0] for n in predecessors]
            branches = [ nx.ancestors(graph,n).union(set([n])) for n in predecessors ] 
            nmax = np.argmax([len(x) for x in branches])            
            nodes_to_keep = branches.pop(nmax)#.union(set(predecessors.pop(nmax)))
            nodes_to_delete = set(n for b in branches for n in b )
            nodes_to_delete = nodes_to_delete - set(nodes_to_keep)
            #nodes_to_keep = nx.ancestors(graph,predecessors[0]).union({predecessors[0]})
            # for pred in predecessors:
            #    nodes_to_delete |= nx.ancestors(graph,pred).union({pred}) - nodes_to_keep
    if nodes_to_delete:
        print(f'deleting {len(nodes_to_delete)} nodes')
        graph.remove_nodes_from(nodes_to_delete)
    # do we have a tree ?
    assert nx.is_branching(graph)




def get_tree_from_root(graph,root_node): 
    # third attempt:
    # - remove cycles 
    # - keep only one ancestor branches in case of multiple ones

    nodes = nx.descendants(graph,root_node) | {root_node}
    if not nodes:
        raise Exception("No root node?!")
    # the subgraph 
    subg = graph.subgraph(nodes).copy()
    remove_cycles(subg)
    assert nx.is_branching(subg)
    return subg 


def get_branches_tree(graph,check_tree=True):
    if not check_tree:
        print("Warning, won't check if we return trees")
    def get_branches(tree,n):
        #succs = list(tree.successors(n)) 
        allsuccs = [n]#+succs
        while True:
            succs = list(tree.successors(n))
            if len(succs) == 1:
                n = succs[0]
                allsuccs.append(n)    
            else:
                allsuccs.append(n)
                break
        allsuccs = [allsuccs]
        if len(succs) > 1:
            for succ in succs:
                allsuccs +=  get_branches(tree,succ)
        return allsuccs
    #graph = combine_trees(trees)
    # find root nodes
    root_nodes = [x for x in graph.nodes() if graph.in_degree(x)==0]
    nodes_lists = []
    for root_node in root_nodes:
        tree = graph.subgraph(nx.descendants(graph,root_node) | {root_node})
        if check_tree: assert nx.is_branching(tree)
        nodes_lists += get_branches(tree,root_node)
    return nodes_lists


def get_all_children_paths(graph,check_tree=True):
    # start from the end 
    leave_nodes = [x for x in graph.nodes() if graph.out_degree(x)==0]
    nodes_lists = []
    for leaf_node in leave_nodes:
        branches = get_path_to_root(leaf_node,graph)
        nodes_lists.append(branches)
    return nodes_lists



def align_times(graph,spots,fluo_step = 18,debug=False):
    ''' This function aligns the graph and spot timepoints to those of frames so that they can be matched.
        This assumes that the previous spot location still corresponds  to the location of the cell when fluorescence is taken
    ''' 
    from copy import deepcopy
    graph = deepcopy(graph)
    spots = spots.copy()
    for n in graph.nodes:
        t = graph.nodes[n]['t'] 
        # if I'm not aligned
        if t % fluo_step != 0:
            pred = list(graph.predecessors(n))
            if not pred:
                continue
            pred = pred[0] 
            frame = t // fluo_step
            t_pred = graph.nodes[pred]['t']
            frame_pred = t_pred   // fluo_step
            # and my predecessor is not aligned
            if frame_pred == frame -1 and frame_pred %18 != 0:
                # them, I update the frame of the predecessor to match mine (it is assumed that the cell 
                t_req = frame*fluo_step
                graph.nodes[pred]['t'] = t_req
                spots.loc[pred,'t'] = t_req 
                if debug: print(f'{pred} reindex {t_pred} --> {t_req}')
    return graph,spots 

def match_spots(spots,lab):
    from functools import partial
    def find_closest(lab,spot): 
        # take slice of data corresponding to spot 
        lab = lab[lab.t == spot.t]
        if lab.empty:
            return np.nan,np.nan
        # distance between all labels
        ds = (spot.x-lab.x)**2 + (spot.y-lab.y)**2
        closest = ds.idxmin()
        min_dist = np.sqrt(ds.loc[closest])
        return closest,min_dist
    
    fn = partial(find_closest,lab)
    match = spots.apply(fn,axis=1)
    spots = spots.copy()
    # lab_id corresponds to index of the matched label in the lab dataframe
    spots['lab_id'],spots['min_dist'] = zip(*match)
    # select spots for which there is fluorescence data 
    matched_spots = spots[spots['lab_id'].notna()]
    # retrieve the indices in the lab dataframe
    idx_label_spot = matched_spots['lab_id'].astype(int)
    # retreive the labels data corresponding to each spot, and reindex it
    labels_reindexed = lab.loc[idx_label_spot].set_index(matched_spots.index.values).drop(columns=['x','y','t'])
    # now add this extra data to the labels
    spots = pd.concat((spots,labels_reindexed),axis=1)
    return spots

def add_fluo_to_graph(graph,spots,lab_df):
    ''' adds to the networkX graph the properties extracted from the segmented cells finding matching spots
        in the "spots" dataframe '''
    newspots = match_spots(spots,lab_df)
    graph = graph.copy()
    for idx,spot in newspots.dropna().iterrows():
        if graph.nodes.get(idx):
            graph.nodes[idx].update(spot.to_dict())
    return graph,newspots
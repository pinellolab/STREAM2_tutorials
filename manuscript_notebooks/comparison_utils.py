import sklearn
import sys
import pandas as pd
import scanpy as sc
import numpy as np
import pyVIA.core as via
import networkx as nx
import igraph as ig
import pygam as pg
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KernelDensity, NearestNeighbors

def get_trajectory_gams(X_dimred, sc_supercluster_nn, cluster_labels, super_cluster_labels, super_edgelist, x_lazy,
                         alpha_teleport,
                         projected_sc_pt, true_label, knn, ncomp, final_super_terminal, sub_terminal_clusters,
                         title_str="hitting times", super_root=[0], draw_all_curves = True,arrow_width_scale_factor=15):
    #X_dimred=X_dimred*1./np.max(X_dimred, axis=0)
    x = X_dimred[:, 0]
    y = X_dimred[:, 1]
    max_x = np.percentile(x, 90)
    noise0 = max_x / 1000

    df = pd.DataFrame({'x': x, 'y': y, 'cluster': cluster_labels, 'super_cluster': super_cluster_labels,
                       'projected_sc_pt': projected_sc_pt},
                      columns=['x', 'y', 'cluster', 'super_cluster', 'projected_sc_pt'])
    df_mean = df.groupby('cluster', as_index=False).mean()
    sub_cluster_isin_supercluster = df_mean[['cluster', 'super_cluster']]

    sub_cluster_isin_supercluster = sub_cluster_isin_supercluster.sort_values(by='cluster')
    sub_cluster_isin_supercluster['int_supercluster'] = sub_cluster_isin_supercluster['super_cluster'].round(0).astype(
        int)

    df_super_mean = df.groupby('super_cluster', as_index=False).mean()

    pt = df_super_mean['projected_sc_pt'].values
    pt_int = [int(i) for i in pt]
    pt_str = [str(i) for i in pt_int]
    pt_sub = [str(int(i)) for i in df_mean['projected_sc_pt'].values]

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=[20, 10])
    num_true_group = len(set(true_label))
    num_cluster = len(set(super_cluster_labels))
    line = np.linspace(0, 1, num_true_group)
    for color, group in zip(line, sorted(set(true_label))):
        where = np.where(np.array(true_label) == group)[0]
        ax1.scatter(X_dimred[where, 0], X_dimred[where, 1], label=group, c=np.asarray(plt.cm.jet(color)).reshape(-1, 4),
                    alpha=0.5, s=10)  # 0.5 and 4
    ax1.legend(fontsize=6)
    ax1.set_title('true labels, ncomps:' + str(ncomp) + '. knn:' + str(knn))

    G_orange = ig.Graph(n=num_cluster, edges=super_edgelist)
    ll_ = [] #this can be activated if you intend to simplify the curves
    for fst_i in final_super_terminal:
        #print('draw traj gams:', G_orange.get_shortest_paths(super_root[0], to=fst_i))
        path_orange = G_orange.get_shortest_paths(super_root[0], to=fst_i)[0]
        len_path_orange = len(path_orange)
        for enum_edge, edge_fst in enumerate(path_orange):
            if enum_edge < (len_path_orange - 1):
                ll_.append((edge_fst, path_orange[enum_edge + 1]))

    if draw_all_curves == True:  edges_to_draw = super_edgelist #default is drawing all super-edges
    else: edges_to_draw = list(set(ll_))
    out = []
    for e_i, (start, end) in enumerate(edges_to_draw):  # enumerate(list(set(ll_))):# : use the ll_ if you want to simplify the curves
    #for e_i, (start, end) in enumerate(list(set(ll_))):# : use the ll_ if you want to simplify the curves

        if pt[start] >= pt[end]:
            temp = end
            end = start
            start = temp

        x_i_start = df[df['super_cluster'] == start]['x'].values
        y_i_start = df[df['super_cluster'] == start]['y'].values
        x_i_end = df[df['super_cluster'] == end]['x'].values
        y_i_end = df[df['super_cluster'] == end]['y'].values
        direction_arrow = 1

        super_start_x = X_dimred[sc_supercluster_nn[start], 0]
        super_end_x = X_dimred[sc_supercluster_nn[end], 0]
        super_start_y = X_dimred[sc_supercluster_nn[start], 1]
        super_end_y = X_dimred[sc_supercluster_nn[end], 1]
        if super_start_x > super_end_x: direction_arrow = -1
        ext_maxx = False
        minx = min(super_start_x, super_end_x)
        maxx = max(super_start_x, super_end_x)

        miny = min(super_start_y, super_end_y)
        maxy = max(super_start_y, super_end_y)

        x_val = np.concatenate([x_i_start, x_i_end])
        y_val = np.concatenate([y_i_start, y_i_end])

        idx_keep = np.where((x_val <= maxx) & (x_val >= minx))[
            0]
        idy_keep = np.where((y_val <= maxy) & (y_val >= miny))[
            0]

        idx_keep = np.intersect1d(idy_keep, idx_keep)

        x_val = x_val[idx_keep]
        y_val = y_val[idx_keep]

        super_mid_x = (super_start_x + super_end_x) / 2
        super_mid_y = (super_start_y + super_end_y) / 2
        from scipy.spatial import distance

        very_straight = False
        if abs(minx - maxx) <= 1:
            very_straight = True
            straight_level = 10
            noise = noise0
            x_super = np.array(
                [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x, super_end_x, super_start_x,
                 super_end_x, super_start_x + noise, super_end_x + noise,
                 super_start_x - noise, super_end_x - noise, super_mid_x])
            y_super = np.array(
                [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y, super_end_y, super_start_y,
                 super_end_y, super_start_y + noise, super_end_y + noise,
                 super_start_y - noise, super_end_y - noise, super_mid_y])
        else:
            straight_level = 3
            noise = noise0
            x_super = np.array(
                [super_start_x, super_end_x, super_start_x, super_end_x, super_start_x, super_end_x, super_start_x,
                 super_end_x, super_start_x + noise, super_end_x + noise,
                 super_start_x - noise, super_end_x - noise])
            y_super = np.array(
                [super_start_y, super_end_y, super_start_y, super_end_y, super_start_y, super_end_y, super_start_y,
                 super_end_y, super_start_y + noise, super_end_y + noise,
                 super_start_y - noise, super_end_y - noise])

        for i in range(straight_level):  # DO THE SAME FOR A MIDPOINT TOO
            y_super = np.concatenate([y_super, y_super])
            x_super = np.concatenate([x_super, x_super])

        list_selected_clus = list(zip(x_val, y_val))

        if (len(list_selected_clus) >= 1) & (very_straight == True):

            dist = distance.cdist([(super_mid_x, super_mid_y)], list_selected_clus, 'euclidean')

            if len(list_selected_clus) >= 2:
                k = 2
            else:
                k = 1
            midpoint_loc = dist[0].argsort()[:k]

            midpoint_xy = []
            for i in range(k):
                midpoint_xy.append(list_selected_clus[midpoint_loc[i]])

            noise = noise0 * 2

            if k == 1:
                mid_x = np.array([midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][
                    0] - noise])
                mid_y = np.array([midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][
                    1] - noise])
            if k == 2:
                mid_x = np.array(
                    [midpoint_xy[0][0], midpoint_xy[0][0] + noise, midpoint_xy[0][0] - noise, midpoint_xy[1][0],
                     midpoint_xy[1][0] + noise, midpoint_xy[1][0] - noise])
                mid_y = np.array(
                    [midpoint_xy[0][1], midpoint_xy[0][1] + noise, midpoint_xy[0][1] - noise, midpoint_xy[1][1],
                     midpoint_xy[1][1] + noise, midpoint_xy[1][1] - noise])
            for i in range(3):
                mid_x = np.concatenate([mid_x, mid_x])
                mid_y = np.concatenate([mid_y, mid_y])

            x_super = np.concatenate([x_super, mid_x])
            y_super = np.concatenate([y_super, mid_y])
        x_val = np.concatenate([x_val, x_super])
        y_val = np.concatenate([y_val, y_super])

        x_val = x_val.reshape((len(x_val), -1))
        y_val = y_val.reshape((len(y_val), -1))
        xp = np.linspace(minx, maxx, 500)

        gam50 = pg.LinearGAM(n_splines=4, spline_order=3, lam=10).gridsearch(x_val, y_val)

        XX = gam50.generate_X_grid(term=0, n=500)

        preds = gam50.predict(XX)

        if ext_maxx == False:
            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]
        else:
            idx_keep = np.where((xp <= (maxx)) & (xp >= (minx)))[0]


        ax2.plot(XX, preds, linewidth=3.5, c='#323538')#1.5
        
        mean_temp = np.mean(xp[idx_keep])
        closest_val = xp[idx_keep][0]
        closest_loc = idx_keep[0]

        for i, xp_val in enumerate(xp[idx_keep]):

            if abs(xp_val - mean_temp) < abs(closest_val - mean_temp):
                closest_val = xp_val
                closest_loc = idx_keep[i]
        step = 1

        head_width = noise * arrow_width_scale_factor  #arrow_width needs to be adjusted sometimes # 40#30  ##0.2 #0.05 for mESC #0.00001 (#for 2MORGAN and others) # 0.5#1
        if direction_arrow == 1:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc + step] - xp[closest_loc],
                      preds[closest_loc + step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')  # , head_starts_at_zero = direction_arrow )
            out.append([XX,preds,direction_arrow,head_width,xp[closest_loc], preds[closest_loc], xp[closest_loc + step] - xp[closest_loc],
                      preds[closest_loc + step] - preds[closest_loc]])

        else:
            ax2.arrow(xp[closest_loc], preds[closest_loc], xp[closest_loc - step] - xp[closest_loc],
                      preds[closest_loc - step] - preds[closest_loc], shape='full', lw=0, length_includes_head=False,
                      head_width=head_width, color='#323538')  # dimgray head_width=head_width
            out.append([XX,preds,direction_arrow,head_width,xp[closest_loc], preds[closest_loc], xp[closest_loc - step] - xp[closest_loc],
                      preds[closest_loc - step] - preds[closest_loc]])            

    c_edge = []
    width_edge = []
    pen_color = []
    super_cluster_label = []
    terminal_count_ = 0
    dot_size = []

    for i in sc_supercluster_nn:
        if i in final_super_terminal:
            print('super cluster', i, 'is a super terminal with sub_terminal cluster',
                  sub_terminal_clusters[terminal_count_])
            width_edge.append(2)
            c_edge.append('yellow')  # ('yellow')
            pen_color.append('black')
            super_cluster_label.append('TS' + str(sub_terminal_clusters[terminal_count_]))  # +'('+str(i)+')')
            dot_size.append(60)
            terminal_count_ = terminal_count_ + 1
        else:
            width_edge.append(0)
            c_edge.append('black')
            pen_color.append('red')
            super_cluster_label.append(str(' '))  # i
            dot_size.append(00)  # 20

    ax2.set_title('lazy:' + str(x_lazy) + ' teleport' + str(alpha_teleport) + 'super_knn:' + str(knn))
    ax2.scatter(X_dimred[:, 0], X_dimred[:, 1], c=projected_sc_pt, cmap='viridis_r', alpha=0.8, s=12)  # alpha=0.6, s=10
    count_ = 0
    loci = [sc_supercluster_nn[key] for key in sc_supercluster_nn]
    for i, c, w, pc, dsz in zip(loci, c_edge, width_edge, pen_color, dot_size):  # sc_supercluster_nn
        ax2.scatter(X_dimred[i, 0], X_dimred[i, 1], c='black', s=dsz, edgecolors=c, linewidth=w)
        count_ = count_ + 1

    plt.title(title_str)
    
    return out, projected_sc_pt, (loci, c_edge, width_edge, pen_color, dot_size)


def run_VIA(input_data,embedding,labels=None, root_user = [0], v0_too_big = 0.3 , v1_too_big = 0.1, v0_random_seed = 42, knn = 20,ncomps=20,jac_std_global=.15,dist_std_local=1):
    if labels==None:
        labels=np.ones(len(input_data))
    v0 = via.VIA(input_data[:, 0:ncomps],labels,root_user=root_user, do_impute_bool=True, is_coarse=True, 
                 preserve_disconnected=False, jac_std_global=jac_std_global, dist_std_local=dist_std_local, knn=knn, too_big_factor=v0_too_big, 
                 random_seed=v0_random_seed) 
    v0.run_VIA()

    tsi_list = via.get_loc_terminal_states(v0, input_data)
    v1 = via.VIA(input_data[:, 0:ncomps], labels, root_user=root_user,too_big_factor=v1_too_big, knn=knn, super_cluster_labels=v0.labels, super_node_degree_list=v0.node_degree_list, super_terminal_cells=tsi_list, is_coarse=False, full_neighbor_array=v0.full_neighbor_array,
                 full_distance_array=v0.full_distance_array, ig_full_graph=v0.ig_full_graph,
                 csr_array_locally_pruned=v0.csr_array_locally_pruned,
                 preserve_disconnected=False,
                 super_terminal_clusters=v0.terminal_clusters, random_seed=v0_random_seed)

    v1.run_VIA()

    super_clus_ds_PCA_loc = via.sc_loc_ofsuperCluster_PCAspace(v0, v1, np.arange(0, len(v1.labels))) #location in PCA space of terminal state

    via_out = get_trajectory_gams(embedding, super_clus_ds_PCA_loc, v1.labels, v0.labels, v0.edgelist_maxout,
                             v1.x_lazy, v1.alpha_teleport, v1.single_cell_pt_markov, labels, knn=v0.knn,
                             final_super_terminal=v1.revised_super_terminal_clusters,
                             sub_terminal_clusters=v1.terminal_clusters,
                             title_str='Markov Hitting Times and Paths', ncomp=ncomps, arrow_width_scale_factor=30)
    if embedding.shape[1]>2:
        via_out2 = get_trajectory_gams(embedding[:,[0,2]], super_clus_ds_PCA_loc, v1.labels, v0.labels, v0.edgelist_maxout,
                                 v1.x_lazy, v1.alpha_teleport, v1.single_cell_pt_markov, labels, knn=v0.knn,
                                 final_super_terminal=v1.revised_super_terminal_clusters,
                                 sub_terminal_clusters=v1.terminal_clusters,
                                 title_str='Markov Hitting Times and Paths', ncomp=ncomps, arrow_width_scale_factor=30)
    
        via_out3 = get_trajectory_gams(embedding[:,[1,2]], super_clus_ds_PCA_loc, v1.labels, v0.labels, v0.edgelist_maxout,
                                 v1.x_lazy, v1.alpha_teleport, v1.single_cell_pt_markov, labels, knn=v0.knn,
                                 final_super_terminal=v1.revised_super_terminal_clusters,
                                 sub_terminal_clusters=v1.terminal_clusters,
                                 title_str='Markov Hitting Times and Paths', ncomp=ncomps, arrow_width_scale_factor=30)
        return via_out, via_out2, via_out3
    else:
        return via_out, None, None

def run_VIA_disconnected(input_data,embedding,labels=None, root_user = [0],
                         knn=20,jac_std_global=0.15,dataset='',random_seed=42,v0_toobig=0.3,
                         marker_genes=[],ncomps=20,preserve_disconnected=True,cluster_graph_pruning_std=0.15,):
    
    if labels==None:
        labels=np.ones(len(input_data))
    v0 = via.VIA(input_data[:, 0:ncomps], labels, jac_std_global=jac_std_global, dist_std_local=1, knn=knn,
             too_big_factor=v0_toobig,  root_user=root_user, dataset=dataset, random_seed=random_seed,
             do_impute_bool=True, is_coarse=True, preserve_disconnected=preserve_disconnected,cluster_graph_pruning_std = cluster_graph_pruning_std) 
    v0.run_VIA()

    #plot coarse cluster heatmap
    if len(marker_genes)>0:
        adata.obs['via0'] = [str(i) for i in v0.labels]
        sc.pl.matrixplot(adata, marker_genes, groupby='via0', dendrogram=True)
        plt.show()

    # get knn-graph and locations of terminal states in the embedded space
    super_clus_ds_PCA_loc = via.sc_loc_ofsuperCluster_PCAspace(v0, v0, np.arange(0, len(v0.labels)))

    via_out = get_trajectory_gams(embedding, super_clus_ds_PCA_loc, v0.labels, v0.labels, v0.edgelist_maxout,
                             v0.x_lazy, v0.alpha_teleport, v0.single_cell_pt_markov, labels, knn=v0.knn,
                             final_super_terminal=v0.terminal_clusters,
                             sub_terminal_clusters=v0.terminal_clusters,
                             title_str='Markov Hitting Times and Paths', ncomp=ncomps, arrow_width_scale_factor=30)
    if embedding.shape[1]>2:
        via_out2 = get_trajectory_gams(embedding[:,[0,2]], super_clus_ds_PCA_loc, v0.labels, v0.labels, v0.edgelist_maxout,
                                 v0.x_lazy, v0.alpha_teleport, v0.single_cell_pt_markov, labels, knn=v0.knn,
                                 final_super_terminal=v0.terminal_clusters,
                                 sub_terminal_clusters=v0.terminal_clusters,
                                 title_str='Markov Hitting Times and Paths', ncomp=ncomps, arrow_width_scale_factor=30)

        via_out3 = get_trajectory_gams(embedding[:,[1,2]], super_clus_ds_PCA_loc, v0.labels, v0.labels, v0.edgelist_maxout,
                                 v0.x_lazy, v0.alpha_teleport, v0.single_cell_pt_markov, labels, knn=v0.knn,
                                 final_super_terminal=v0.terminal_clusters,
                                 sub_terminal_clusters=v0.terminal_clusters,
                                 title_str='Markov Hitting Times and Paths', ncomp=ncomps, arrow_width_scale_factor=30)
        return via_out, via_out2, via_out3
    else:
        return via_out, None, None

def run_paga(_anndata,X,resolution=1,use_rep='X_dr'):
    ###paga
    sc.pp.neighbors(_anndata,use_rep=use_rep)
    sc.tl.leiden(_anndata,resolution)
    sc.tl.paga(_anndata)

    paga_edges = np.argwhere(_anndata.uns['paga']['connectivities'])
    clus=np.array(_anndata.obs['leiden']).astype(int)
    u_clus=np.unique(clus)
    clus_NodePositions = []
    for i,c in enumerate(u_clus):
        clus_NodePositions.append(X[clus==c].mean(axis=0))
    paga_nodep=np.array(clus_NodePositions)
    paga_weights=np.array(_anndata.uns['paga']['connectivities'][_anndata.uns['paga']['connectivities'].nonzero()])[0]
    return paga_nodep, paga_edges, paga_weights
#import stream as st
import anndata
import numpy as np
import networkx as nx
import scipy
import scanpy as sc
import pickle
import seaborn as sns
import elpigraph
import sklearn.metrics
import pandas as pd
import ot
import wot
import plotly.express as px
import matplotlib.pyplot as plt
import umap
import KDEpy 
import sys; sys.path.append('../final_notebooks/')
#import elpigraph_v2
#import utils
import scipy as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors
from plotly.subplots import make_subplots

def geodesic_pseudotime(X,k,root):
    '''pseudotime as graph distance from root point'''
    nn = NearestNeighbors(n_neighbors=k,n_jobs=8).fit(X)
    g = nx.convert_matrix.from_scipy_sparse_matrix(nn.kneighbors_graph(mode='distance'))
    if len(list(nx.connected_components(g))) > 1:
        raise ValueError(f'detected more than 1 components with k={k} neighbors. Please increase k')
    lengths = nx.single_source_dijkstra_path_length(g,root)
    pseudotime=np.array(pd.Series(lengths).sort_index())
    return pseudotime

def make_plotly_2d(adatas,data,names,color,color_discrete_map,fpath):    
    #---------labels
    fig  = make_subplots(2,2,horizontal_spacing=0.05,vertical_spacing=0.05,subplot_titles=names,shared_xaxes=True,shared_yaxes=True)

    axs = np.array(np.meshgrid([1,2],[1,2])).T.reshape(-1,2)
    for i,_adata in enumerate(adatas):
        n_labels=len(np.unique(_adata.obs[color]))
        fig.add_trace(px.scatter(x=min_max(_adata.obsm[data])[0],y=min_max(_adata.obsm[data])[1],opacity=0).data[0],row=[axs[i][0]],col=[axs[i][1]])
        fig.add_traces(px.scatter(x=_adata.obsm[data][:,0],y=_adata.obsm[data][:,1],color_discrete_map=color_discrete_map,
                                  color=_adata.obs[color]).data,
                                  rows=[axs[i][0]]*n_labels,
                                  cols=[axs[i][1]]*n_labels)
        if i < 3: 
            for t in range(-n_labels,0): fig.data[t].showlegend=False

    fig.update_layout(width=1300,height=1300) \
       .update_traces(marker_size=3,marker_opacity=.6) \
       .write_html(fpath)

def make_plotly_2d_label(adatas,data,names,color,color_discrete_map,fpath):    
    #---------labels
    fig  = make_subplots(2,2,horizontal_spacing=0.05,vertical_spacing=0.05,subplot_titles=names,shared_xaxes=True,shared_yaxes=True)

    axs = np.array(np.meshgrid([1,2],[1,2])).T.reshape(-1,2)
    for i,_adata in enumerate(adatas):
        n_labels=len(np.unique(_adata.obs[color]))
        fig.add_trace(px.scatter(x=min_max(_adata.obsm[data])[0],y=min_max(_adata.obsm[data])[1],opacity=0).data[0],row=[axs[i][0]],col=[axs[i][1]])
        fig.add_traces(px.scatter(x=_adata.obsm[data][:,0],y=_adata.obsm[data][:,1],color_discrete_map=color_discrete_map,
                                  color=_adata.obs[color]).data,
                                  rows=[axs[i][0]]*n_labels,
                                  cols=[axs[i][1]]*n_labels)
        #if i < 3: 
        #    for t in range(-n_labels,0): fig.data[t].showlegend=False

    fig.update_layout(width=1300,height=1300) \
       .update_traces(marker_size=3,marker_opacity=.6) \
       .write_html(fpath) 
    
def make_plotly_2d_continuous(data,names,color,color_discrete_map,fpath):    
    #---------labels
    fig  = make_subplots(2,2,horizontal_spacing=0.05,vertical_spacing=0.05,subplot_titles=names,shared_xaxes=True,shared_yaxes=True)

    axs = np.array(np.meshgrid([1,2],[1,2])).T.reshape(-1,2)
    for i,_adata in enumerate(adatas):
        fig.add_trace(px.scatter(x=_adata.obsm[data][:,0],y=_adata.obsm[data][:,1],
                                  color=_adata.uns['detailed_gene_score_df'][color]).data[0],
                                  row=axs[i][0],
                                  col=axs[i][1])
    fig.update_layout(width=1300,height=1300) \
       .update_traces(marker_size=3,marker_opacity=.6) \
       .write_html(fpath)
    
def get_marker_df_PA():
    df_PA=pd.DataFrame(pd.Series(
        {'Memory B-cell': 'B-cells',
        'Naive B-cell': 'B-cells',
        'Myeloid DC': 'Dendritic cells',
        'Plasmacytoid DC': 'Dendritic cells',
        'Basophil': 'Granulocytes',
        'Eosinophil': 'Granulocytes',
        'Neutrophil': 'Granulocytes',
        'Classical monocyte': 'Monocytes',
        'Intermediate monocyte': 'Monocytes',
        'Non-classical monocyte': 'Monocytes',
        'NK-cell': 'NK-cells',
        'GdT-cell': 'T-cells',
        'MAIT T-cell': 'T-cells',
        'Memory CD4 T-cell': 'T-cells',
        'Memory CD8 T-cell': 'T-cells',
        'Naive CD4 T-cell': 'T-cells',
        'Naive CD8 T-cell': 'T-cells',
        'T-reg': 'T-cells'}
    )).reset_index()
    df_PA.columns=['Cell type','Cell lineage']
    df_PA['fname']=['memory_B','naive_B','myeloid','plasmacytoid','basophil_Cell','eosinophil_Cell','neutrophil_Cell','classical','intermediate','non-classical','NK-cell_Cell','gdT-cell_Cell','MAIT','memory_CD4','memory_CD8','naive_CD4','naive_CD8','T-reg_Cell',]

    markers_PA = {} ; states = []
    for i in range(len(df_PA['fname'])):
        n=df_PA['fname'][i] ; ct=df_PA['Cell type'][i]
        markers_PA[ct] = pd.read_json(f'../data/hematopoiesis/markers/Protein_atlas_cell_types/blood_cell_category_rna_{n}.json')['Gene']
        states.append(np.repeat(ct,len(markers_PA[ct])))
    states = np.concatenate(states)

    detailed_markers = pd.DataFrame({'Symbol':np.concatenate(list(markers_PA.values())),'Cell State':states})

    markers = detailed_markers['Symbol'].unique()
    cell_types = detailed_markers['Cell State'].unique()

    marker_df = pd.DataFrame(0,index=markers,columns=cell_types)
    for ct in cell_types:
        idx_ct = detailed_markers.loc[detailed_markers['Cell State']==ct,'Symbol']
        marker_df.loc[idx_ct,ct] = 1

    #--- plotting variables
    color_discrete_map_celltypes = {k:v for k,v in  zip(np.append(cell_types,'undefined'),px.colors.qualitative.Dark24[:len(cell_types)]+['lightgrey'])}
    
    return detailed_markers, markers, cell_types, marker_df, color_discrete_map_celltypes

def get_marker_df_HCA_main(min_pearson_correlation=-np.inf, max_pval=np.inf):
    detailed_markers = pd.read_csv('../data/hematopoiesis/markers/Pellin_et_al_paper/Cell_type_markers.csv')
    detailed_markers = detailed_markers[(detailed_markers['Pearson rho'] > min_pearson_correlation) & (detailed_markers['Pearson p-value'] < max_pval)]
    detailed_markers['Cell State.1'][detailed_markers['Symbol']=='CD14']='Monocyte'

    markers = detailed_markers['Symbol'].unique()
    cell_types = detailed_markers['Cell State.1'].unique()

    marker_df = pd.DataFrame(0,index=markers,columns=cell_types)
    for ct in cell_types:
        idx_ct = detailed_markers.loc[detailed_markers['Cell State.1']==ct,'Symbol']
        marker_df.loc[idx_ct,ct] = 1

    #--- plotting variables
    color_discrete_map_celltypes = {k:v for k,v in  zip(np.append(cell_types,'undefined'),px.colors.qualitative.Dark24[:len(cell_types)]+['lightgrey'])}
    return detailed_markers, markers, cell_types, marker_df, color_discrete_map_celltypes

def get_marker_df_HCA_all(min_pearson_correlation=-np.inf, max_pval=np.inf):
    detailed_markers = pd.read_excel('../data/hematopoiesis/markers/Markers-Centroids-Cells-HCA-35-populations.xlsx')
    detailed_markers = detailed_markers[(detailed_markers['Pearson rho'] > min_pearson_correlation) & (detailed_markers['Pearson p-value'] < max_pval)]
    detailed_markers['Cell State'][detailed_markers['Symbol']=='CD14']='Monocyte'

    markers = detailed_markers['Symbol'].unique()
    cell_types = detailed_markers['Cell State'].unique()

    marker_df = pd.DataFrame(0,index=markers,columns=cell_types)
    for ct in cell_types:
        idx_ct = detailed_markers.loc[detailed_markers['Cell State']==ct,'Symbol']
        marker_df.loc[idx_ct,ct] = 1

    #--- plotting variables
    color_discrete_map_celltypes = {k:v for k,v in  zip(np.append(cell_types,'undefined'),px.colors.qualitative.Dark24[:len(cell_types)]+['lightgrey'])}
    return detailed_markers, markers, cell_types, marker_df, color_discrete_map_celltypes

def get_main_markers():
    HSC_markers = ["SPINK2", "CRHBP", "MEIS1", "MLLT3"]
    MEP_markers = ["GATA1", "HBD", "TFRC", "UROD", "NFIA", "KLF1"]
    Myeloid_markers = ["MPO", "CEBPA", "ELANE", "IRF8", "LGALS1"]
    Lymphoid_markers = ["DNTT", "CXCR4", "CD79A", "BLNK", "IGLL1", "EBF1"]
    Yurino_markers = ["TPSB2", "CPA3", "RNASE2", "CLC", "MPO", "CXCR2", "LYZ", "MAFB", "SLC7A7", "CD1C", "FLT3", "IRF7", "TNFRSF21"]

    markers_dict = {'HSC':HSC_markers,'MEP':MEP_markers,'Myeloid':Myeloid_markers,'Lymphoid':Lymphoid_markers,'Yurino':Yurino_markers}

    markers_main = np.array(list(set(HSC_markers+MEP_markers+Myeloid_markers+Lymphoid_markers+Yurino_markers)))
    cell_types_main = markers_dict.keys()

    marker_df_main = pd.DataFrame(0,index=markers_main,columns=cell_types_main)
    for ct in cell_types_main:
        marker_df_main.loc[markers_dict[ct],ct] = 1
    
    return markers_dict, cell_types_main, marker_df_main

def get_marker_lineage_tracing():
    xl = pd.ExcelFile('../data/hematopoiesis/markers/lineage_tracing_paper/diff_expr_genes in_vitro_mature_mono_Neu-like_Mo-like.xlsx')
    df_diff_vitro_mature=xl.parse(xl.sheet_names)
    xl = pd.ExcelFile('../data/hematopoiesis/markers/lineage_tracing_paper/diff_expr_genes in_vitro_day2_mono_Neu-like_Mo-like.xlsx')
    df_diff_vitro_day2=xl.parse(xl.sheet_names)
    xl = pd.ExcelFile('../data/hematopoiesis/markers/lineage_tracing_paper/diff_expr_genes in_vivo_mature_mono_Neu-like_Mo-like.xlsx')
    df_diff_vivo=xl.parse(xl.sheet_names)

    df=pd.read_excel('../data/hematopoiesis/markers/lineage_tracing_paper/cell_types_markers.xlsx')
    for i in range(len(df)):
        if df['Cell type'][[i]].isnull().any(): 
            df['Cell type'][i] = df['Cell type'][max(i-1,0)] 
            df['Abbreviation '][i] = df['Abbreviation '][max(i-1,0)] 
    df_cell_types=df[~df['Marker gene'].isnull()].copy()
    df_cell_types.rename(columns={'Marker gene':'Symbol','Cell type':'Cell State.1'},inplace=1)

    df_cell_types_early=pd.read_excel('../data/hematopoiesis/markers/lineage_tracing_paper/cell_types_markers_early_commitment.xlsx')
    df_cell_types_early.rename(columns={'Gene symbol':'Symbol','Lineage':'Cell State.1'},inplace=1)
    return df_diff_vitro_mature, df_diff_vitro_day2, df_diff_vivo, df_cell_types, df_cell_types_early

def annotate_adatas(names, adatas, detailed_markers, permutations = 0,percentile = 99,ct_key='Cell State.1'):
    cell_types = detailed_markers[ct_key].unique()
    
    for name,_anndata in zip(names,adatas):
        print(name)
        
        #---cells x markers boolean matrix
        gs = sc.AnnData(np.zeros((_anndata.shape[1],len(cell_types))))
        gs.obs_names = _anndata.var_names
        gs.var_names = cell_types
        for ct in cell_types:
            found_markers = np.isin(gs.obs_names, detailed_markers[detailed_markers[ct_key]==ct]['Symbol'])
            gs[found_markers, ct] = 1

        #---gene score df from gs boolean matrix
        detailed_gene_set_scores_df = pd.DataFrame(index=_anndata.obs.index)
        for j in range(gs.shape[1]):
            gene_set_name = str(gs.var.index.values[j])
            result = wot.score_gene_sets(ds=_anndata, gs=gs[:, [j]], permutations=permutations, method='mean_z_score')
            detailed_gene_set_scores_df[gene_set_name + '_score'] = result['score']
            if permutations > 0:
                detailed_gene_set_scores_df[gene_set_name + '_p_value'] = result['p_value']
                detailed_gene_set_scores_df[gene_set_name + '_fdr_bh'] = result['fdr']
                detailed_gene_set_scores_df[gene_set_name + '_k'] = result['k']

        #detailed_gene_set_scores_df.to_csv(f'../results/gene_set_scores_{name}.csv', index_label='id')
        detailed_gene_set_scores_adata= sc.AnnData(X=detailed_gene_set_scores_df.values, obs=_anndata.obs, var=pd.DataFrame(index=detailed_gene_set_scores_df.columns))
        column_filter = list(filter(lambda x: '_score' in x, detailed_gene_set_scores_adata.var_names))
        _anndata.uns['detailed_gene_score_df']=detailed_gene_set_scores_df[column_filter]


        #---gene scores to annotation from max gene score
        _anndata.obs['cell_ids_from_max']=_anndata.uns['detailed_gene_score_df'].idxmax(axis=1).map(lambda s: s[:-6])

        #---gene scores to annotation from 99% percentile and significant p value
        cell_set_name_to_ids = {}
        gs_adata = detailed_gene_set_scores_adata[:,column_filter]
        percentiles = np.percentile(gs_adata.X, axis=0, q=percentile)

        _anndata.uns['cell_ids']=detailed_gene_set_scores_df[column_filter].copy()
        _anndata.uns['cell_ids'][:]=False
        if len(percentiles.shape) == 0: percentiles = [percentiles]

        for j in range(gs_adata.shape[1]):  # each gene set
            x = gs_adata[:, j].X
            selected = x >= percentiles[j]
            if permutations > 0:
                selected_pval = detailed_gene_set_scores_adata[:,gs_adata.var_names[j][:-6]+'_p_value'].X < 0.05
                cell_ids = gs_adata[selected & selected_pval].obs.index
            else:
                cell_ids = gs_adata[selected].obs.index
            if len(cell_ids) > 0:
                cell_set_name_to_ids[gs_adata.var.index[j]] = cell_ids
                _anndata.uns['cell_ids'].loc[selected.flat,gs_adata.var.index[j]] = True
        _anndata.uns['cell_ids'].columns = gs.var.index
        #_anndata.uns['cell_ids'].to_csv(f'../results/cell_types_from_gene_score_{name}.csv')

        idx_dup = _anndata.uns['cell_ids'].sum(axis=1) > 1
        idx_nan = _anndata.uns['cell_ids'].sum(axis=1) == 0
        _anndata.obs['cell_ids_from_quantile']=_anndata.uns['cell_ids'].idxmax(axis=1)
        _anndata.obs['cell_ids_from_quantile'][idx_dup] = _anndata.obs['cell_ids_from_max'][idx_dup]
        _anndata.obs['cell_ids_from_quantile'][idx_nan] = 'undefined'


def get_M_matrices(adatas,key):
    annots=[np.array(a.obs[key]).astype(str) for a in adatas]
    cell_types=np.unique(np.concatenate(annots))
    cell_types=cell_types[cell_types!='undefined']
    
    M = []
    n_datasets = len(annots)
    for k in range(n_datasets-1):
        M.append(np.ones((len(annots[k]), len(annots[-1]))))
        for ct in cell_types:
            ct_xs = np.where(annots[k]==ct)[0]
            ct_xt = np.where(annots[-1]==ct)[0]
            for i in ct_xs:
                M[k][i,ct_xt] = 0.1
    return M

def qGW(C1,C2,p1,p2,sample_size = .5):
    samples = int(sample_size*C1.shape[0])
    node_subset1 = list(set(quantizedGW.sample(list(range(C1.shape[0])),samples)))
    node_subset2 = list(set(quantizedGW.sample(list(range(C2.shape[0])),samples)))
    coup_qgw = quantizedGW.compressed_gw_point_cloud(C1,C2,p1,p2,node_subset1,node_subset2,verbose = True,return_dense = True)
    return coup_qgw

def run_qGW(xs,xt,p1=None,p2=None,sample_size=.5):
    ''' Returns qGW mapping'''
    #---- setup inputs
    if p1 is None: p1 = ot.unif(C1.shape[0]) 
    if p2 is None: p2 = ot.unif(C2.shape[0]) 
    #C1 = scipy.spatial.distance.cdist(xs, xs)
    #C2 = scipy.spatial.distance.cdist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()
    
    #---- qGW mapping
    coupling=qGW(C1,C2, p1, p2, sample_size = sample_size)
    transp = coupling / np.sum(coupling, 1)[:, None]
    transp[~ np.isfinite(transp)] = 0
    xstq = transp @ xt
     
    return xs_aligned, xstq

def kNN(X,k):
    '''get graph distances'''
    nn = NearestNeighbors(n_neighbors=k,n_jobs=8).fit(X)
    g = nn.kneighbors_graph(mode='distance')
    return g

def weight_per_label(xs_labels, yt_labels):
    # Weighting points by label proportion, so that
    # - Label total weights is equal in each dataset
    # - Dataset-specific labels weight 0

    n, m = len(xs_labels), len(yt_labels)
    all_labels = list(set(xs_labels).union(set(yt_labels)))

    # Labels specific to xs/yt
    labels_specx = [ i for (i, li) in enumerate(all_labels)
                     if li not in yt_labels ]
    labels_specy = [ i for (i, li) in enumerate(all_labels)
                     if li not in xs_labels ]
    labels_common = [ i for (i, li) in enumerate(all_labels)
                      if i not in labels_specx
                      and i not in labels_specy ]

    # Fequency of each label
    xs_freqs = np.array([
        np.sum(xs_labels == li) / n for li in all_labels
    ])
    yt_freqs = np.array([
        np.sum(yt_labels == li) / m for li in all_labels
    ])

    # Only accounting for common labels
    norm_x, norm_y = (
        np.sum(xs_freqs[labels_common]),
        np.sum(yt_freqs[labels_common])
    )
    rel_freqs = np.zeros(len(all_labels))
    rel_freqs[labels_common] = (
        yt_freqs[labels_common] * norm_x / (xs_freqs[labels_common] * norm_y)
    )

    # Correcting weights with respect to label frequency
    wx, wy = np.ones(n) / n, np.ones(m) / m
    for fi, li in zip(rel_freqs, all_labels):
        wx[xs_labels == li] *= fi

    return wx / wx.sum(), wy / wy.sum()


def get_graph_distance_matrix(data, num_neighbors, mode="distance", metric="cosine"):
    """
    Compute graph distance matrices on data 
    """
    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode=="connectivity":
        include_self=True
    else:
        include_self=False
    graph_data=kneighbors_graph(data, num_neighbors, mode=mode, metric=metric, include_self=include_self)
    shortestPath_data= dijkstra(csgraph= csr_matrix(graph_data), directed=False, return_predecessors=False)
    shortestPath_max= np.nanmax(shortestPath_data[shortestPath_data != np.inf])
    shortestPath_data[shortestPath_data > shortestPath_max] = shortestPath_max
    shortestPath_data=shortestPath_data/shortestPath_data.max()

    return shortestPath_data



def weight_per_label(xs_labels, yt_labels):
    # Weighting points by label proportion, so that
    # - Label total weights is equal in each dataset
    # - Dataset-specific labels weight 0

    n, m = len(xs_labels), len(yt_labels)
    all_labels = list(set(xs_labels).union(set(yt_labels)))

    # Labels specific to xs/yt
    labels_specx = [ i for (i, li) in enumerate(all_labels)
                     if li not in yt_labels ]
    labels_specy = [ i for (i, li) in enumerate(all_labels)
                     if li not in xs_labels ]
    labels_common = [ i for (i, li) in enumerate(all_labels)
                      if i not in labels_specx
                      and i not in labels_specy ]

    # Fequency of each label
    xs_freqs = np.array([
        np.sum(xs_labels == li) / n for li in all_labels
    ])
    yt_freqs = np.array([
        np.sum(yt_labels == li) / m for li in all_labels
    ])

    # Only accounting for common labels
    norm_x, norm_y = (
        np.sum(xs_freqs[labels_common]),
        np.sum(yt_freqs[labels_common])
    )
    rel_freqs = np.zeros(len(all_labels))
    rel_freqs[labels_common] = (
        yt_freqs[labels_common] * norm_x / (xs_freqs[labels_common] * norm_y)
    )

    # Correcting weights with respect to label frequency
    wx, wy = np.ones(n) / n, np.ones(m) / m
    for fi, li in zip(rel_freqs, all_labels):
        wx[xs_labels == li] *= fi
    for i in labels_specx + labels_specy:
        wy[yt_labels == all_labels[i]] = 0

    return wx / wx.sum(), wy / wy.sum()

def get_M_matrices(adatas,key,strength=.5):
    annots=[np.array(a.obs[key]).astype(str) for a in adatas]
    cell_types=np.unique(np.concatenate(annots))
    cell_types=cell_types[cell_types!='undefined']
    
    M = []
    n_datasets = len(annots)
    for k in range(n_datasets-1):
        M.append(np.ones((len(annots[k]), len(annots[-1]))))
        for ct in cell_types:
            ct_xs = np.where(annots[k]==ct)[0]
            ct_xt = np.where(annots[-1]==ct)[0]
            for i in ct_xs:
                M[k][i,ct_xt] = strength
    return M

def make_plotly_2d_continuous(adatas,data,names,color,color_discrete_map,fpath):    
    #---------labels
    fig  = make_subplots(2,2,horizontal_spacing=0.05,vertical_spacing=0.05,subplot_titles=names,shared_xaxes=True,shared_yaxes=True)

    axs = np.array(np.meshgrid([1,2],[1,2])).T.reshape(-1,2)
    for i,_adata in enumerate(adatas):
        try:
            c=_adata[:,color.upper()].X.copy().toarray().flatten()
        except:
            c=_adata[:,color.upper()].X.copy().flatten()
        fig.add_trace(px.scatter(x=_adata.obsm[data][:,0],y=_adata.obsm[data][:,1],
                                  color=c).data[0],
                                  row=axs[i][0],
                                  col=axs[i][1])
    fig.update_layout(width=1300,height=1300) \
       .update_traces(marker_size=3,marker_opacity=.6) \
       .write_html(fpath)
    
    
def make_plotly_2d_integrated(adata,data,names,color,color_discrete_map,fpath):    
    #---------labels
    fig  = make_subplots(2,2,horizontal_spacing=0.05,vertical_spacing=0.05,subplot_titles=names,
                         shared_xaxes=True,shared_yaxes=True)

    axs = np.array(np.meshgrid([1,2],[1,2])).T.reshape(-1,2)
    for i,n in enumerate(names):
        _adata = adata[adata.obs['assay_detailed']==n]
        n_labels=len(np.unique(_adata.obs[color]))
        fig.add_trace(px.scatter(x=min_max(_adata.obsm[data])[0],y=min_max(_adata.obsm[data])[1],opacity=0).data[0],row=[axs[i][0]],col=[axs[i][1]])
        fig.add_traces(px.scatter(x=_adata.obsm[data][:,0],y=_adata.obsm[data][:,1],color_discrete_map=color_discrete_map,
                                  color=_adata.obs[color]).data,
                                  rows=[axs[i][0]]*n_labels,
                                  cols=[axs[i][1]]*n_labels)
        if i < 3: 
            for t in range(-n_labels,0): fig.data[t].showlegend=False

    fig.update_layout(width=1300,height=1300,legend_itemsizing='constant',font=dict(size=25)) \
       .update_traces(marker_size=3,marker_opacity=.6) \
       .update_annotations(font_size=30) \
       .write_html(fpath)
    
def make_plotly_2d_integrated_label(adata,data,names,color,color_discrete_map,fpath):    
    #---------labels
    fig  = make_subplots(2,2,horizontal_spacing=0.05,vertical_spacing=0.05,subplot_titles=names,shared_xaxes=True,shared_yaxes=True)

    axs = np.array(np.meshgrid([1,2],[1,2])).T.reshape(-1,2)
    for i,n in enumerate(names):
        _adata = adata[adata.obs['assay_detailed']==n[6:]]
        n_labels=len(np.unique(_adata.obs[color]))
        fig.add_trace(px.scatter(x=min_max(_adata.obsm[data])[0],y=min_max(_adata.obsm[data])[1],opacity=0).data[0],row=[axs[i][0]],col=[axs[i][1]])
        fig.add_traces(px.scatter(x=_adata.obsm[data][:,0],y=_adata.obsm[data][:,1],color_discrete_map=color_discrete_map,
                                  color=_adata.obs[color]).data,
                                  rows=[axs[i][0]]*n_labels,
                                  cols=[axs[i][1]]*n_labels)
        #if i < 3: 
        #    for t in range(-n_labels,0): fig.data[t].showlegend=False

    fig.update_layout(width=1300,height=1300,legend_itemsizing='constant',font=dict(size=25)) \
       .update_traces(marker_size=3,marker_opacity=.6) \
       .update_annotations(font_size=30) \
       .write_html(fpath)
    
def make_plotly_barplots():
    import plotly.graph_objects as go
    
    cell_types=np.unique(np.concatenate([a.obs['leiden_identity2'] for a in adatas]))
    color_discrete_map_celltypes = {k:v for k,v in  zip(np.append(cell_types,'undefined'),px.colors.qualitative.Dark24[:len(cell_types)]+['lightgrey'])}
    #---------labels
    fig  = make_subplots(2,2,horizontal_spacing=0.05,vertical_spacing=0.05,subplot_titles=names,shared_xaxes='all',shared_yaxes='all')

    axs = np.array(np.meshgrid([1,2],[1,2])).T.reshape(-1,2)
    for i,_adata in enumerate(adatas):

        counts = [sum(_adata.obs['leiden_identity2']==c) for c in cell_types]
        c = [color_discrete_map_celltypes[c] for c in cell_types]
        fig.add_trace(go.Bar(x=counts,y=cell_types,marker_color=c,orientation='h'),
                          row=[axs[i][0]],
                          col=[axs[i][1]])


    fig.update_layout(width=1300,height=1300) \
       .update_traces() \
       .write_html('bar.html') 


#---markers variables/utils
min_max = lambda X : ((X[:,0].min()*1.05,X[:,0].max()*1.05), (X[:,1].min()*1.05,X[:,1].max()*1.05))
labels = ['0', '1', '3', '5',  '6','7','8', '9','11', '12','CD34posCD33neg', 'CD34posCD33dim', 'CD33pos', 'CD34negCD33dim']
color_discrete_map_labels = {k:v for k,v in  zip(labels,px.colors.qualitative.Dark24)}




from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from d2v_func import *

sample_df = pd.read_csv('sample_df_for_training.csv',index_col=0)
model= Doc2Vec.load("d2v.model")

model1_pvecs = get_perspective_vectors(model, 100)
model1_svecs = get_source_vectors(model, sample_df, 100)
model1_avecs = get_all_vectors_labels(model, 100,sample_df)
all_vecs = model1_avecs.append(model1_svecs)
all_vecs = all_vecs.append(model1_pvecs)

def PCA_modeling(all_vectors, num_columns):
    #fit vectors to PCA
    scaler = StandardScaler()
    data_std = scaler.fit_transform(all_vecs)
    data_std = pd.DataFrame(data_std)
    data_std.columns = list(model1_pvecs)
    pca = PCA(n_components=num_columns)
    pca.fit(data_std)
    #plot explained variance
    x_values = range(1, pca.n_components_+1)
    explained_variance_ratio = pca.explained_variance_
    plot_explained_variance(x_values,explained_variance_ratio)
    #plot 3 Principal Components
    PC_df = pd.DataFrame(pca.transform(data_std), index=all_vecs.index)
    return PC_df


def PCA_plot(PCA_DF): #trace options [articles,sources,perspectives]
    all_sources = ['The New York Times','MSNBC','Fox News','The Wall Street Journal','The American Conservative','Breitbart News','Time','CNN','National Review','Daily Mail','Vice News','Associated Press','The Economist','Reuters','The Washington Times']
    all_perspectives = ['right', 'left', 'center']
    s_df = PCA_DF.loc[all_sources]
    p_df = PCA_DF.loc[all_perspectives]
    all_labels = all_sources
    all_labels.extend(all_perspectives)
    a_df = PCA_DF.drop(all_labels)

    articles = go.Scatter3d(
        x=a_df[0],
        y=a_df[1],
        z=a_df[2],
        mode='markers',
        text=a_df.index,
        marker=dict(
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.4
        )
    )

    sources = go.Scatter3d(
        x=s_df[0],
        y=s_df[1],
        z=s_df[2],
        mode='markers',
        text=s_df.index,
        marker=dict(
            size=20,
            line=dict(
                color='green',
                width=0.5
            ),
            opacity=0.8
        )
    )

    perspectives = go.Scatter3d(
        x=p_df[0],
        y=p_df[1],
        z=p_df[2],
        mode='markers',
        text=p_df.index,
        marker=dict(
            size=20,
            line=dict(
                color='red',
                width=0.5
            ),
            opacity=0.8
        )
    )



    data = [sources,articles,perspectives]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename='plot from API (5)')

def plot_explained_variance(total_features,exp_variance_ratio):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(total_features, exp_variance_ratio, lw=2, label='explained variance')
    ax.plot(total_features, np.cumsum(exp_variance_ratio), lw=2, label='cumulative explained variance')
    ax.set_title('Doc2vec: explained variance of components')
    ax.set_xlabel('principal component')
    ax.set_ylabel('explained variance')
    plt.show()

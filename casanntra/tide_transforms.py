import pandas as pd
from sklearn.decomposition import PCA

def difference(x):
    return x.diff().bfill()

def append_tide_diff(df):
    df = df.copy()
    df['sf_dsub'] = df.groupby('case')['sf_tidal_filter'].transform(difference)
    return df


def append_tidal_pca_cols(df):
    data_normalized = df[['sf_tidal_filter', 'sf_tidal_energy','case']] 
    data_normalized.loc[:,"sf_tidal_filter"] -= data_normalized.groupby("case")["sf_tidal_filter"].transform("mean")
    data_normalized.loc[:,"sf_tidal_filter"] /= data_normalized.sf_tidal_filter.std()
    data_normalized.loc[:,"sf_tidal_energy"] -=  data_normalized.sf_tidal_energy.mean()  # Note global for energy, group for filter
    data_normalized.loc[:,"sf_tidal_energy"] /= data_normalized.sf_tidal_energy.std()
    data_normalized = data_normalized.drop("case",axis=1)


    # Perform PCA
    pca = PCA(n_components=2)
    pca_loadings = pca.fit_transform(data_normalized)
    print(data_normalized.columns)
    pca_df = pd.DataFrame(index=data_normalized.index,
                        data={'tidal_pc1': pca_loadings[:, 0],
                        'tidal_pc2': pca_loadings[:, 1]})
    df_out = pd.concat([df,pca_df],axis=1)
    return df_out



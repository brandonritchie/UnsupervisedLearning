#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Churn Data
churn = pd.read_csv('https://raw.githubusercontent.com/brandonritchie/SupervisedLearningProject1/master/Bank%20Customer%20Churn%20Prediction.csv')
# Data cleaning
cleaned_churn = churn.drop(['customer_id'], axis = 1)
# Encode for models later on
churn = pd.get_dummies(cleaned_churn, columns = ['country', 'gender'])
X = churn.drop(['churn'], axis = 1)
sc = StandardScaler()
churn_scaled = pd.DataFrame(sc.fit_transform(X.values), columns = X.columns)

##############
data_raw = pd.read_csv('https://raw.githubusercontent.com/brandonritchie/UnsupervisedLearning/main/breast-cancer.csv')
data_raw['diagnosis'] = np.where(data_raw['diagnosis'] == 'M', 1, 0)
data_raw2 = data_raw.drop(columns = 'id')
y = data_raw2[['diagnosis']]
X = data_raw2.drop(columns = 'diagnosis')

sc2 = StandardScaler()
cs_scaled = pd.DataFrame(sc2.fit_transform(X.values), columns = X.columns)

#%%
# CLUSTERING - Step 1

## K-Means - Churn
# https://realpython.com/k-means-clustering-python/
def get_kmeans(data):
    ss_l = []
    k_l = []
    i_l = []
    for k in range(2,15):
        print(k)
        kmeans = KMeans(
            init="random",
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=1984
        )
        kmeans.fit(data)
        l=kmeans.predict(data)
        ss = silhouette_score(data, l)
        i_l.append(kmeans.inertia_)
        k_l.append(k)
        ss_l.append(ss)
    
    return(ss_l, k_l, i_l)

ss_churn, k_churn, i_churn = get_kmeans(churn_scaled)
ss_cs, k_cs, i_cs = get_kmeans(cs_scaled)


# Notes:
'''
The optimal number of clusters appears to be 6. There is a slight elbow
formed from the intertia plot. Based on the sillhouette score there
it is quite noisy for the first few clusters but it performs the best on 6 clusters.
'''
fig = plt.gcf()
fig.set_size_inches(6, 6)
# Sillhouette score
plt.title('K-Means Sillhoutte Score by Clusters - Churn')
plt.plot(k_churn, ss_churn)
plt.show()
# Inertia scores
plt.title('K-Means Inertia by Clusters - Churn')
plt.plot(k_churn, i_churn)
plt.show()

# Notes:
'''
This dataset does not appear to have distinct clustering patterns due to a lack of
an elbow in the elbow plot and a small sillhouette score. The best performing 
cluster appears to be 2 because this maximizes the sillhouette score.
'''
## K-Means Product Subscription
# Sillhouette score
fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.title('K- Means Sillhoutte Score by Clusters - Breast Cancer')
plt.plot(k_cs, ss_cs)
plt.show()
# Inertia scores
plt.title('K-Means Inertia by Clusters - Breast Cancer')
plt.plot(k_cs, i_cs)
plt.show()

#%%
# Evaluate K-Means Clustering
import seaborn as sns
kmeans_churn = KMeans(
            init="random",
            n_clusters=3,
            n_init=10,
            max_iter=300,
            random_state=1984
        ).fit(churn_scaled)
churn_plot = churn_scaled[['tenure', 'estimated_salary','gender_Male']]
churn_plot['cluster'] = kmeans_churn.predict(churn_scaled)
churn['cluster'] = kmeans_churn.predict(churn_scaled)
sns.pairplot(churn_plot, hue = 'cluster')
# Gender barplot
g_dat = churn.value_counts(['gender_Male', 'cluster']).reset_index().rename(columns = {0:'Count', 'gender_Male':'gender'})
g_dat['gender'] = np.where(g_dat['gender'] == 1, 'Male', 'Female')
sns.catplot(x = "gender",
            y = 'Count',
            hue = "cluster",
            kind = 'bar',
            data = g_dat,
            legend = False)
ax = plt.gca()
ax.legend(fontsize = 12,
          title = "Cluster #",
          title_fontsize = 12)

# Breast Cancer
kmeans_cs = KMeans(
            init="random",
            n_clusters=2,
            n_init=10,
            max_iter=300,
            random_state=1984
        ).fit(cs_scaled)

cs_plot = cs_scaled[['perimeter_mean', 'compactness_mean', 'concavity_mean']]
cs_plot['cluster'] = kmeans_cs.predict(cs_scaled)
sns.pairplot(cs_plot, hue = 'cluster')
data_raw3 = data_raw2[['perimeter_mean', 'compactness_mean', 'concavity_mean', 'diagnosis']]
sns.pairplot(data_raw3, hue = 'diagnosis')

#%%
## Expectation Maximization - Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
def get_gm(data):
    ss_l = []
    k_l = []
    for k in range(2,15):
        print(k)
        gm = GaussianMixture(n_components=k, 
        random_state=1984).fit(data)
        l = gm.predict(data)
        ss = silhouette_score(data, l)
        k_l.append(k)
        ss_l.append(ss)
    
    return(ss_l, k_l)

ss_churn_em, k_churn_em = get_gm(churn_scaled)
ss_cs_em, k_cs_em = get_gm(cs_scaled)

fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.title('EM Sillhoutte Score by Clusters - Churn')
plt.plot(k_churn_em, ss_churn_em)
plt.show()

fig = plt.gcf()
fig.set_size_inches(6, 6)
plt.title('EM Sillhoutte Score by Clusters - Customer Segmentation')
plt.plot(k_cs_em, ss_cs_em)
plt.show()

#%%
# Evaluate Expectation Maximization
em_churn = GaussianMixture(
            n_components=3, 
            random_state=1984
        ).fit(churn_scaled)
churn_plot = churn_scaled[['tenure', 'estimated_salary', 'gender_Male']]
churn_plot['cluster'] = em_churn.predict(churn_scaled)
churn_plot['churn'] = churn['churn']
sns.pairplot(churn_plot, hue = 'cluster')

churn_plot2 = pd.concat([churn_plot[churn.churn == 1], churn_plot[churn.churn == 0].sample(n = 2037)])
clus0_1 = np.array(churn_plot2[churn_plot2.cluster == 0]['churn']).sum() / len(churn_plot2[churn_plot2.cluster == 0])
clus0_0 = 1 - clus0_1
clus1_1 = np.array(churn_plot2[churn_plot2.cluster == 1]['churn']).sum() / len(churn_plot2[churn_plot2.cluster == 1])
clus1_0 = 1 - clus1_1
clus2_1 = np.array(churn_plot2[churn_plot2.cluster == 2]['churn']).sum() / len(churn_plot2[churn_plot2.cluster == 2])
clus2_0 = 1 - clus0_1
cluster_l = [0,0,1,1,2,2]
label_l = [1,0,1,0,1,0]
prop_l = [clus0_1,clus0_0,clus1_1,clus1_0,clus2_1,clus2_0]
dat_plot5 = pd.DataFrame({'cluster':cluster_l, 'true_label':label_l, 'proportion':prop_l}).query('true_label == 1').sort_values('proportion', ascending = False)
b = sns.barplot(data = dat_plot5, x = 'cluster', y = 'proportion')
b.set_xlabel("Cluster",fontsize=20)
b.set_ylabel("Value",fontsize=20)
b.tick_params(labelsize=20)
g_dat = churn_plot.value_counts(['gender_Male', 'cluster']).reset_index().rename(columns = {0:'Count', 'gender_Male':'gender'})
g_dat['gender'] = np.where(g_dat['gender'] >0, 'Male', 'Female')
sns.catplot(x = "gender",
            y = 'Count',
            hue = "cluster",
            kind = 'bar',
            data = g_dat,
            legend = False)
ax = plt.gca()
ax.legend(fontsize = 12,
          title = "Cluster #",
          title_fontsize = 12)


em_cs = GaussianMixture(
            n_components=2, 
            random_state=1984
        ).fit(cs_scaled)
cs_plot = cs_scaled[['perimeter_mean', 'compactness_mean', 'concavity_mean']]
cs_plot['cluster'] = em_cs.predict(cs_scaled)
sns.pairplot(cs_plot, hue = 'cluster')

# %%
# DIMENSIONALITY REDUCTION - Part 2
## PCA
import numpy as np
from sklearn.decomposition import PCA
def pca_var(data):
    pca = PCA(n_components=10)
    pca.fit(data)
    variance = list(pca.explained_variance_ratio_)
    while True:
        variance.pop(-1)
        if np.array(variance).sum() < .90:
            break
    return(variance)

def skree_plot(data, name, variance):
    pcas = ['PCA ' + str(i + 1) for i in range(len(variance))]
    d = pd.DataFrame({'PCA':pcas, 'ExplainedVariance':variance})
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.title(f'PCA Skree Plot - {name}')
    plt.bar(d.PCA, d.ExplainedVariance)
    plt.ylabel('Explained Variance')


skree_plot(churn_scaled, 'Churn', pca_var(churn_scaled))

skree_plot(cs_scaled, 'Breast Cancer', pca_var(cs_scaled))
# %%
# Evaluate PCA
comps = 3
pca_cs = PCA(n_components=comps).fit_transform(cs_scaled)
pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
pca_df = pd.DataFrame(pca_cs, columns = pca_n)
pca_df['diagnosis'] = data_raw2['diagnosis']
sns.pairplot(pca_df, hue = 'diagnosis')

########
comps = 3
pca_churn = PCA(n_components=comps).fit_transform(churn_scaled)
pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
pca_df = pd.DataFrame(pca_churn, columns = pca_n)
pca_df['churn'] = churn['churn']
sns.pairplot(pca_df, hue = 'churn')
# %%
## ICA
from sklearn.decomposition import FastICA
import scipy

def get_ICA(data, thresh):
    comps = len(data.columns)
    ica = FastICA(n_components=comps).fit_transform(data)
    ica_n = ['ICA ' + str(i + 1) for i in range(comps)]
    ica_vecs = pd.DataFrame(ica, columns = ica_n)

    kurtosis_thresh = thresh
    ICA_cols = []
    ICA_kurtosis = []
    num = 1
    for c in ica_vecs.columns:
        ICA_kurtosis.append(abs(scipy.stats.kurtosis(ica_vecs[c])))
        ICA_cols.append(c)
        num += 1
    kurt_keep = (pd.DataFrame({'ICA':ICA_cols,'Kurtosis':ICA_kurtosis})
        .query(f'Kurtosis > {kurtosis_thresh}')
        .sort_values('Kurtosis', ascending = False))

    ICA_keep = list(kurt_keep.ICA.values)
    ICA_df = ica_vecs[ICA_keep]
    return(ICA_df)

ICA_cs = get_ICA(cs_scaled, 40)

ICA_churn = get_ICA(churn_scaled, 1.5)

#%%
# Evaluate ICA
ICA_cs['diagnosis'] = data_raw2['diagnosis']
sns.pairplot(ICA_cs[['ICA 20', 'ICA 22', 'ICA 7', 'diagnosis']], hue = 'diagnosis')

ICA_churn['churn'] = churn['churn']
sns.pairplot(ICA_churn, hue = 'churn')
# %%
# Randomized Projections
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import mean_squared_error

def random_components_graph(data, name):
    comps = []
    rmse_l = []
    for i in range(2,15):
        comps.append(i)
        rp = GaussianRandomProjection(n_components = i,random_state=182)
        rp_trans = rp.fit_transform(data)

        rp_inverse_trans = rp.inverse_transform(rp_trans)
        rmse = mean_squared_error(data, rp_inverse_trans, squared = False)
        rmse_l.append(rmse)
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.title(f'RMSE Randomized Projections by Number of Components - {name}')
    plt.plot(comps, rmse_l)
    plt.show()

random_components_graph(churn_scaled, 'Churn')
random_components_graph(cs_scaled, 'Breast Cancer')

# %%
# Random Projection Evaluation
rp = GaussianRandomProjection(n_components = 3,random_state=182)
rp_trans = pd.DataFrame(rp.fit_transform(cs_scaled))
rp_trans['diagnosis'] = data_raw2['diagnosis']
fig = plt.gcf()
fig.set_size_inches(6, 6)
rel = sns.pairplot(rp_trans, hue = 'diagnosis')
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle('Random Projection Seperation of Classifier - Breast Cancer')

rp = GaussianRandomProjection(n_components = 3,random_state=182)
rp_trans = pd.DataFrame(rp.fit_transform(churn_scaled))
rp_trans['churn'] = churn['churn']
fig = plt.gcf()
fig.set_size_inches(6, 6)
rel = sns.pairplot(rp_trans, hue = 'churn')
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle('Random Projection Seperation of Classifier - Churn')
# %%
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit(churn_scaled, churn['churn'])

data_plot = pd.DataFrame(lda_fit.transform(churn_scaled), columns = ['LDA'])
data_plot['churn'] = churn['churn']
fig = plt.gcf()
fig.set_size_inches(6, 6)
sns.histplot(data = data_plot, x = 'LDA', hue = 'churn').set(title = 'LDA Seperation - Churn')

accuracy_score(churn['churn'], lda_fit.predict(churn_scaled))

lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit(cs_scaled, data_raw2['diagnosis'])

data_plot = pd.DataFrame(lda_fit.transform(cs_scaled), columns = ['LDA'])
data_plot['diagnosis'] = data_raw2['diagnosis']
fig = plt.gcf()
fig.set_size_inches(6, 6)
sns.histplot(data = data_plot, x = 'LDA', hue = 'diagnosis').set(title = 'LDA Seperation - Breast Cancer')

accuracy_score(data_raw2['diagnosis'], lda_fit.predict(cs_scaled))

#%%
# Clustering on each reduced dataset

# K-means - PCA - Breast Cancer
comps = 1
pca_cs = PCA(n_components=comps).fit_transform(cs_scaled)
pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
pca_df = pd.DataFrame(pca_cs, columns = pca_n)

kmeans_cs = KMeans(
            init="random",
            n_clusters=2,
            n_init=10,
            max_iter=300,
            random_state=1984
        ).fit(pca_df)

pca_df['cluster'] = kmeans_cs.predict(pca_df)
pca_df['diagnosis'] = data_raw2['diagnosis']

rel = sns.pairplot(pca_df, hue = 'diagnosis')
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle('Kmeans Clustering - PCA - Breast Cancer')

# K-means - ICA - Breast Cancer
ICA_cs = get_ICA(cs_scaled, 100)
kmeans_cs = KMeans(
            init="random",
            n_clusters=2,
            n_init=10,
            max_iter=300,
            random_state=1984
        ).fit(ICA_cs)

ICA_cs['cluster'] = kmeans_cs.predict(ICA_cs)
ICA_cs['diagnosis'] = data_raw2['diagnosis']
sns.pairplot(ICA_cs, hue = 'diagnosis')

# K-means - Randomized Projection - Breast Cancer
rp = GaussianRandomProjection(n_components = 2,random_state=182)
rp_trans = pd.DataFrame(rp.fit_transform(cs_scaled), columns = ['Projection 1', 'Projection 2'])
kmeans_cs = KMeans(
            init="random",
            n_clusters=2,
            n_init=10,
            max_iter=300,
            random_state=1984
        ).fit(rp_trans)
rp_trans['cluster'] = kmeans_cs.predict(rp_trans)
rp_trans['diagnosis'] = data_raw2['diagnosis']
rel = sns.pairplot(rp_trans, hue = 'diagnosis')
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle('Kmeans Clustering - PCA - Breast Cancer')

# Expectation Maximization - PCA - Breast Cancer
comps = 1
pca_cs = PCA(n_components=comps).fit_transform(cs_scaled)
pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
pca_df = pd.DataFrame(pca_cs, columns = pca_n)

em_cs = GaussianMixture(
            n_components=2, 
            random_state=1984
        ).fit(pca_df)
pca_df['cluster'] = em_cs.predict(pca_df)
pca_df['diagnosis'] = data_raw2['diagnosis']
rel = sns.pairplot(pca_df, hue = 'diagnosis')
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle('EM Clustering - PCA - Breast Cancer')

# Expectation Maximization - ICA - Breast Cancer
ICA_cs = get_ICA(cs_scaled, 100)
em_cs = GaussianMixture(
            n_components=2, 
            random_state=1984
        ).fit(ICA_cs)
ICA_cs['cluster'] = em_cs.predict(ICA_cs)
ICA_cs['diagnosis'] = data_raw2['diagnosis']
sns.pairplot(ICA_cs, hue = 'diagnosis')

# Expectation Maximization - Randomized Projections - Breast Cancer
rp = GaussianRandomProjection(n_components = 2,random_state=182)
rp_trans = pd.DataFrame(rp.fit_transform(cs_scaled), columns = ['Projection 1', 'Projection 2'])
em_cs = GaussianMixture(
            n_components=2, 
            random_state=1984
        ).fit(rp_trans)
rp_trans['cluster'] = em_cs.predict(rp_trans)
rp_trans['diagnosis'] = data_raw2['diagnosis']
sns.pairplot(rp_trans, hue = 'diagnosis')

# Expectation Maximization - LDA - Breast Cancer
lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit(cs_scaled, data_raw2['diagnosis'])
lda_cs = pd.DataFrame(lda_fit.transform(cs_scaled), columns = ['LDA'])
em_cs = GaussianMixture(
            n_components=2, 
            random_state=1984
        ).fit(lda_cs)
lda_cs['cluster'] = em_cs.predict(lda_cs)
lda_cs['diagnosis'] = data_raw2['diagnosis']
sns.pairplot(lda_cs[['LDA', 'diagnosis']], hue = 'diagnosis')

# Kmeans - PCA - Churn
comps = 1
pca_churn = PCA(n_components=comps).fit_transform(churn_scaled)
pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
pca_df = pd.DataFrame(pca_churn, columns = pca_n)

kmeans_churn = KMeans(
            init="random",
            n_clusters=2,
            n_init=10,
            max_iter=300,
            random_state=1984
        ).fit(pca_df)

pca_df['cluster'] = kmeans_churn.predict(pca_df)
pca_df['churn'] = churn['churn']
rel = sns.pairplot(pca_df, hue = 'churn')
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle('Kmeans Clustering - PCA - Churn')


# EM - Randomized Projections - Churn
rp = GaussianRandomProjection(n_components = 2,random_state=182)
rp_trans = pd.DataFrame(rp.fit_transform(churn_scaled), columns = ['Projection 1', 'Projection 2'])
em_cs = GaussianMixture(
            n_components=2, 
            random_state=1984
        ).fit(rp_trans)
rp_trans['cluster'] = em_cs.predict(rp_trans)
rp_trans['churn'] = churn['churn']
sns.pairplot(rp_trans, hue = 'churn')

# KMeans - LDA - Churn
lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit(churn_scaled, churn['churn'])
lda_churn = pd.DataFrame(lda_fit.transform(churn_scaled), columns = ['LDA'])
kmeans_churn = KMeans(
            init="random",
            n_clusters=2,
            n_init=10,
            max_iter=300,
            random_state=1984
        ).fit(lda_churn)
lda_churn['cluster'] = kmeans_churn.predict(lda_churn)
lda_churn['churn'] = churn['churn']
sns.pairplot(lda_churn, hue = 'churn')

# EM - PCA - Churn
comps = 3
pca_cs = PCA(n_components=comps).fit_transform(churn_scaled)
pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
pca_df = pd.DataFrame(pca_cs, columns = pca_n)

em_cs = GaussianMixture(
            n_components=2, 
            random_state=1984
        ).fit(pca_df)
pca_df['cluster'] = em_cs.predict(pca_df)
pca_df['churn'] = churn['churn']
sns.pairplot(pca_df, hue = 'churn')
# %%
# Train Neural Network for Churn on all 4 DR algorithms
# Include wall clock times, measure performance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from pytictoc import TicToc
t = TicToc()
p = len(churn[churn.churn == 1])
churn_neg = churn[churn.churn == 0].sample(n = p)
churn_pos = churn[churn.churn == 1]
churn2 = pd.concat([churn_pos,churn_neg])
X = churn2.drop(['churn'], axis = 1)
y = churn2[['churn']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
sc2 = StandardScaler()
X_train = pd.DataFrame(sc2.fit_transform(X_train.values), columns = X_train.columns)
X_test = pd.DataFrame(sc2.transform(X_test.values), columns = X_test.columns)
# PCA
comps = 9
pca_fit = PCA(n_components=comps).fit(X_train)
pca_transform = pca_fit.transform(X_train)
pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
pca_train = pd.DataFrame(pca_transform, columns = pca_n)

parameters = {'activation':('logistic', 'relu'), 'hidden_layer_sizes':[(3,32),(3,64),(3,128)], 'solver':('lbfgs','adam')}
clf = MLPClassifier()
scores = GridSearchCV(clf, parameters)
scores.fit(pca_train, y_train)
scores.best_estimator_
# MLPClassifier(activation='logistic', hidden_layer_sizes=(3, 64), solver='lbfgs')

# ICA
comps = len(X_train.columns)
ica_fit = FastICA(n_components=comps).fit(X_train)
ica_transform = ica_fit.transform(X_train)
ica_n = ['ICA ' + str(i + 1) for i in range(comps)]
ica_vecs = pd.DataFrame(ica_transform, columns = ica_n)

kurtosis_thresh = 1.5
ICA_cols = []
ICA_kurtosis = []
num = 1
for c in ica_vecs.columns:
    ICA_kurtosis.append(abs(scipy.stats.kurtosis(ica_vecs[c])))
    ICA_cols.append(c)
    num += 1
kurt_keep = (pd.DataFrame({'ICA':ICA_cols,'Kurtosis':ICA_kurtosis})
    .query(f'Kurtosis > {kurtosis_thresh}')
    .sort_values('Kurtosis', ascending = False))

ICA_keep = list(kurt_keep.ICA.values)
ICA_train = ica_vecs[ICA_keep]

parameters = {'activation':('logistic', 'relu'), 'hidden_layer_sizes':[(3,32),(3,64),(3,128)], 'solver':('lbfgs','adam')}
clf = MLPClassifier()
scores = GridSearchCV(clf, parameters)
scores.fit(ICA_train, y_train)
scores.best_estimator_
# MLPClassifier(hidden_layer_sizes=(3, 32), solver='lbfgs')

# Randomized Projections
rp = GaussianRandomProjection(n_components = 2,random_state=182)
rp_fit = rp.fit(X_train)
rp_transform = pd.DataFrame(rp_fit.transform(X_train))

parameters = {'activation':('logistic', 'relu'), 'hidden_layer_sizes':[(3,32),(3,64),(3,128)], 'solver':('lbfgs','adam')}
clf = MLPClassifier()
scores = GridSearchCV(clf, parameters)
scores.fit(rp_trans, y_train)
scores.best_estimator_
# MLPClassifier(hidden_layer_sizes=(3, 32))

# LDA
lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit(X_train, y_train)
lda_dat = pd.DataFrame(lda_fit.transform(X_train), columns = ['LDA'])

parameters = {'activation':('logistic', 'relu'), 'hidden_layer_sizes':[(3,32),(3,64),(3,128)], 'solver':('lbfgs','adam')}
clf = MLPClassifier()
scores = GridSearchCV(clf, parameters)
scores.fit(lda_dat, y_train)
scores.best_estimator_
#MLPClassifier(hidden_layer_sizes=(3, 32), solver='lbfgs')

#%%
# Step 4 - Combine into dataset
value_l = []
metric_l = []
time_l = []
algorithm_l = []

# PCA
t.tic()
clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(3, 64), solver='lbfgs').fit(pca_train, y_train)
time_val = t.tocvalue()
pca_test = pd.DataFrame(pca_fit.transform(X_test), columns = pca_n)
pred = clf.predict(pca_test)
value_l.append(accuracy_score(y_test, pred))
metric_l.append('Accuracy')
time_l.append(time_val)
algorithm_l.append('PCA')
value_l.append(precision_score(y_test, pred))
metric_l.append('Precision')
time_l.append(time_val)
algorithm_l.append('PCA')
value_l.append(recall_score(y_test, pred))
metric_l.append('Recall')
time_l.append(time_val)
algorithm_l.append('PCA')

# ICA
t.tic()
clf = MLPClassifier(hidden_layer_sizes=(3, 32), solver='lbfgs').fit(ICA_train, y_train)
time_val = t.tocvalue()
ica_test = pd.DataFrame(ica_fit.transform(X_test), columns = ica_n)[ICA_keep]
pred = clf.predict(ica_test)
value_l.append(accuracy_score(y_test, pred))
metric_l.append('Accuracy')
time_l.append(time_val)
algorithm_l.append('ICA')
value_l.append(precision_score(y_test, pred))
metric_l.append('Precision')
time_l.append(time_val)
algorithm_l.append('ICA')
value_l.append(recall_score(y_test, pred))
metric_l.append('Recall')
time_l.append(time_val)
algorithm_l.append('ICA')

# Randomized Projection
t.tic()
clf = MLPClassifier(hidden_layer_sizes=(3, 32)).fit(rp_transform, y_train)
time_val = t.tocvalue()
rp_test = pd.DataFrame(rp_fit.transform(X_test))
pred = clf.predict(rp_test)
value_l.append(accuracy_score(y_test, pred))
metric_l.append('Accuracy')
time_l.append(time_val)
algorithm_l.append('RP')
value_l.append(precision_score(y_test, pred))
metric_l.append('Precision')
time_l.append(time_val)
algorithm_l.append('RP')
value_l.append(recall_score(y_test, pred))
metric_l.append('Recall')
time_l.append(time_val)
algorithm_l.append('RP')

# LDA
t.tic()
clf = MLPClassifier(hidden_layer_sizes=(3, 32), solver='lbfgs').fit(lda_dat, y_train)
time_val = t.tocvalue()
lda_test = pd.DataFrame(lda_fit.transform(X_test))
pred = clf.predict(lda_test)
value_l.append(accuracy_score(y_test, pred))
metric_l.append('Accuracy')
time_l.append(time_val)
algorithm_l.append('LDA')
value_l.append(precision_score(y_test, pred))
metric_l.append('Precision')
time_l.append(time_val)
algorithm_l.append('LDA')
value_l.append(recall_score(y_test, pred))
metric_l.append('Recall')
time_l.append(time_val)
algorithm_l.append('LDA')

# Original
t.tic()
clf = MLPClassifier(hidden_layer_sizes=(3,128), activation = 'logistic', solver = 'adam').fit(X_train, y_train)
time_val = t.tocvalue()
pred = clf.predict(X_test)
value_l.append(accuracy_score(y_test, pred))
metric_l.append('Accuracy')
time_l.append(time_val)
algorithm_l.append('Original')
value_l.append(precision_score(y_test, pred))
metric_l.append('Precision')
time_l.append(time_val)
algorithm_l.append('Original')
value_l.append(recall_score(y_test, pred))
metric_l.append('Recall')
time_l.append(time_val)
algorithm_l.append('Original')

result_df = pd.DataFrame({'Algorithm':algorithm_l, 
    'WallClockTime':time_l,
    'Value':value_l,
    'Metric':metric_l}).sort_values(by = 'Value', ascending = False)

sns.set(rc={'figure.figsize':(13,13)})
b = sns.barplot(data = result_df, x = 'Algorithm', y = 'Value', hue = 'Metric')
b.set_xlabel("",fontsize=1)
b.set_ylabel("Value",fontsize=20)
b.tick_params(labelsize=20)
plt.setp(b.get_legend().get_texts(), fontsize='20')
plt.setp(b.get_legend().get_title(), fontsize='0')

wall_clock = result_df[['Algorithm','WallClockTime']].groupby('Algorithm').agg('mean').reset_index().sort_values('WallClockTime', ascending = False)
b = sns.barplot(data = wall_clock, x = 'Algorithm', y = 'WallClockTime')
b.set_xlabel("",fontsize=1)
b.set_ylabel("Seconds to Fit",fontsize=20)
b.tick_params(labelsize=20)
# %%
# Step 5 - Clustering and ANN

# Kmeans PCA churn
def get_pca_kmeans(train, test):
    comps = 9
    pca_fit = PCA(n_components=comps).fit(train)
    pca_transform_train = pca_fit.transform(train)
    pca_transform_test = pca_fit.transform(test)
    pca_n = ['PCA ' + str(i + 1) for i in range(comps)]
    pca_train = pd.DataFrame(pca_transform_train, columns = pca_n)
    pca_test = pd.DataFrame(pca_transform_test, columns = pca_n)
    kmeans_churn = KMeans(
                init="random",
                n_clusters=2,
                n_init=10,
                max_iter=300,
                random_state=1984
            ).fit(pca_train)

    pca_train['cluster'] = kmeans_churn.predict(pca_train)
    pca_test['cluster'] = kmeans_churn.predict(pca_test)
    return(pca_train, pca_test)

pca_train, pca_test = get_pca_kmeans(X_train, X_test)
value_c_l = []
metric_c_l = []
time_c_l = []
algorithm_c_l = []
# Model after grid search
t.tic()
clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(3, 64), solver='lbfgs').fit(pca_train, y_train)
time_val = t.tocvalue()
pred = clf.predict(pca_test)
value_c_l.append(accuracy_score(y_test, pred))
metric_c_l.append('Accuracy')
time_c_l.append(time_val)
algorithm_c_l.append('PCA-KmeansFeature')
value_c_l.append(precision_score(y_test, pred))
metric_c_l.append('Precision')
time_c_l.append(time_val)
algorithm_c_l.append('PCA-KmeansFeature')
value_c_l.append(recall_score(y_test, pred))
metric_c_l.append('Recall')
time_c_l.append(time_val)
algorithm_c_l.append('PCA-KmeansFeature')

result_c_df = pd.DataFrame({'Algorithm':algorithm_c_l, 
    'WallClockTime':time_c_l,
    'Value':value_c_l,
    'Metric':metric_c_l}).sort_values(by = 'Value', ascending = False)

result_df_pca = result_df[result_df.Algorithm == 'PCA']

cluster_compare = pd.concat([result_c_df, result_df_pca])

sns.set(rc={'figure.figsize':(13,13)})
b = sns.barplot(data = cluster_compare, x = 'Algorithm', y = 'Value', hue = 'Metric')
b.set_xlabel("",fontsize=1)
b.set_ylabel("Value",fontsize=20)
b.tick_params(labelsize=20)
plt.setp(b.get_legend().get_texts(), fontsize='20')
plt.setp(b.get_legend().get_title(), fontsize='0')
# %%

##################################################################################################################
### Cristina Izquierdo Lozano (c.izquierdo.lozano@tue.nl)                                                      ###
### Functions to cluster data and create graphical output containing characterization metrics                  ###
### PCA, including feature importance and explained variance                                                   ###
### T-SNE & U-MAP, including the multivariate coefficient of variance (MCV)                                    ###
### Minimum Spanning Tree (MST)                                                                                ###
### Silhouette metric with a generic overview of the clustering and a pairwise comparison of the user's choice ###
### Correlation matrix, including the coefficients of variance (CV) for each feature                           ###
##################################################################################################################

# Import generic packages
import os
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd # To preprocess data and deal with datasets
import numpy as np # Math package
import gc # To clean the memory (garbage collector)
pd.options.mode.chained_assignment = None  # default='warn' --> disables the false positive warning for orverwriting a dataframe when using inplace while droping columns
import re # To match regular expressions

# Packages for plotting figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #For the dummy legends
import seaborn as sns
from adjustText import adjust_text

# Import clustering packages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as UMAP
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples, silhouette_score # CLustering metric

# Import packages for MST
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
import networkx as nx # Plot graph

##################################################################
####################### DATA PREPROCESSING #######################
##################################################################

def importData(path):
    """
    Imports and concats all csv files found in the given path to create the dataset.
    NOTE: Your path folder should only contain the csv meant to be analysed.

    :param path: (string) absolute path to the folder containing *only* your data to be analysed
    
    :return dataset: (pandas dataframe) dataset with the concatenated csv with your data
    """
    filenames = os.listdir(path) #Read filenames in path
    dataset = pd.concat((pd.read_csv(path + '/' + f) for f in filenames), ignore_index=True) #Read and append features csv
    
    return dataset


def npAtlasDataCleaning(data): ## NOTE: EDIT ACCORDING TO YOUR DATASET!!!
    """
    Cleans dataset: drops non-feature columns (X and Y coordinates) and removes the clusters that contain more than 800 localizations.

    :param data: (pandas dataframe) dataset to be cleaned
    
    :return dataset: (pandas dataframe) the cleaned dataset.
    """
     ##################################Drop columns for the NP ATLAS##############################################
    ## Remove non-features columns
    dataset = data.drop(["X coordinate", "Y coordinate"], axis=1)

    ## remove outliers
    dataset = dataset[dataset.loc[:,'Cluster localizations'] < 800] # Remove clusters with more than 800 total localizations

    return dataset


def filterDataset(option, originalDataset):
    """
    Additional quality filters to clean the dataset.

    Option "targetCounts" will remove samples with more targets that localizations (possibly due to a bad fitting)
    Option "fittingQuality" will remove samples where the rSquare value of the fitting, of either of the channels, is lower than 0.8
    Will return a warning if the given option doesn't exist.

    :param option: (string) desired filtering option.
    :param originalDataset: (pandas dataframe) dataset to be cleaned.

    :return dataset: pandas dataframe with the filtered dataset
    """
    match option:
        case 'targetCounts': # NOTE: move this one into the data cleaning because we need this always to filter incorrect data
            dataset = originalDataset[originalDataset['Cluster localizations'] > (originalDataset['channel 1 target count'] + originalDataset['channel 2 target count'])]
        case 'fittingQuality':
            dataset = originalDataset[(originalDataset['channel 2 rSquared true mean dark time'] >= 0.8) & (originalDataset['channel 1 rSquared true mean dark time'] >= 0.8)]
        case _:
            print("Wrong option. Use only \'targetCounts\' or \'fittingQuality\'")
    
    return dataset


def reduceFeatures(dataset): ## NOTE: EDIT ACCORDING TO YOUR DATASET!!
    """
    Reduces the size of the dataset to contain only a few selected features.
    TODO: have features as an input.

    :param dataset: (pandas dataframe) dataset to be reduced.

    :return smaller_dataset: (pandas dataframe) reduced dataset (5 features)
    """
    ## Drop non-relevant columns to print the smaller correlation matrix so the plot is readable (the stats are calculated with the full dataset anyway)
    smaller_dataset = dataset.drop(dataset.iloc[:,3:13], axis=1) # Drop dark time stats columns 
    smaller_dataset = smaller_dataset.drop(smaller_dataset.iloc[:,4:12], axis=1) # Drop bright time stats columns   
   
    return smaller_dataset


def subsampleData(dataset, subFraction, labelHeader):
    """
    Subsample dataset to use in computationally heavy operations (like the MST).

    :param dataset: (pandas dataframe) dataset to be subsampled.
    :param subFraction: (float) fraction of the total dataset that you want to subsample.
    :param labelHeader: (string) header name for the column in your dataset that contains the label to your samples.

    :return subsampledDataset: (pandas dataframe) subsampled dataframe.
    :return subsampledTarget: (string array) your sample labels for the subsampled data.
    """
    ## Subsample the dataset
    subsampledDataset = dataset.groupby(labelHeader).sample(frac=subFraction) # From each sample group, sample subFraction(%) of the nanoparticles
    subsampledTarget = subsampledDataset.loc[:, labelHeader].values #Separating the label column (sample type)

    return subsampledDataset, subsampledTarget


def separateReproducibilityExperiments(dataset, labelHeader): # FOR PANDAS DATAFRAMES
    """
    Merge samples with labels that indicate it's the same formulation but different researcher
    and separate the samples that have been reproduced by different researchers into a separate dataframe.

    :param dataset: (pandas dataframe) original dataset to have the labels merged/separated
    :param labelHeader: (string) header name for the column in your dataset that contains the label to your samples.

    :return reprod_peopleDataset: (pandas dataframe) dataset containing only the samples that have been reproduced by multiple people.
    :return datasetPeopleMerged: (pandas dataframe) original given dataset but renaming the samples that where reproduced to have the same matching label
    """
    ## Separate reproducibility datasets 
    target = dataset.loc[:, labelHeader].values #Separating the label column (sample type)
    reprod_peopleDataset = dataset[(target == '200_medium_VG') | (target == '300_medium_MT') | (target== '300_medium_VG') | (target == '200_medium_MT')]
    datasetPeopleMerged = dataset.replace(regex={r'^200_medium_(MT|VG)': '200_medium', r'^300_medium_(MT|VG)': '300_medium'}) # Rename labels to join them in the analysis

    return reprod_peopleDataset, datasetPeopleMerged


def match_pattern(s, pattern): # Vectorize the pattern
    """
    Quick function to vectorize a regular expression pattern with a given string array. This is meant to be used mainly by the next function (separateExperimentType)

    :param s: (string) element in which we want to find the pattern
    :param pattern: (regular expression) pattern to identify in the given string.

    :return: elements in s that match the pattern
    """
    return re.match(pattern, s) is None


def separateExperimentType(pattern, dataset, target): # FOR NUMPY ARRAYS
    """
    Function to divide the dataset based on a regular expression found in the labels.

    :param pattern: (regular expression) contains the pattern the user wants to base the division on.
    :param dataset: (pandas dataframe) dataset to be divided.
    :param target: (string array) sample labels where the pattern is found

    :return data_notPattern: (numpy array) dataset with the values of the samples that don't belong to the given pattern
    :return data_pattern: (numpy array) dataset with the values of the samples that belong to the given pattern
    :return target_notPattern: (string array) labels for the samples that don't belong to the given pattern
    :return target_pattern: (string array) labels for the samples that belong to the given array
    """
    # Vectorize the function to apply it to each element in the target array
    vectorized_match = np.vectorize(match_pattern)

    # Apply the vectorized function to create a boolean mask
    mask_notPattern = vectorized_match(target, pattern) # Elements that do NOT contain the pattern
    mask_pattern = ~mask_notPattern # Elements that contain the pattern

    # Split the data array using the masks
    data_notPattern = dataset[mask_notPattern]
    data_pattern = dataset[mask_pattern]

    # Split the data array using the masks
    target_notPattern = target[mask_notPattern]
    target_pattern = target[mask_pattern]

    return data_notPattern, data_pattern, target_notPattern, target_pattern


def preprocess4clustering(dataset, labelHeader, mean, minmax):
    """
    Standardize the data for the clustering algorithms and separate the values from the target (sample labels).
    If the minmax flag is on, the data will be normalized using the minmax scaler. (For MST)

    :param dataset: (pandas dataframe) dataset to be scaled
    :param labelHeader: (string) header name for the column in your dataset that contains the label to your samples.
    :param mean: (dictionary of numpy arrays) mean values for each feature for each sample.
    :param minmax: (boolean) flag to calculate the minmax normalization instead of standardization

    :return scaledX: (numpy array) scaled data, with mass centers
    :return target: (string array) separated labels, as in the given dataset
    :return targetMass: (string array) separated labels, with the mass centers added
    :return uniqueLabels: (list) sorted unique sample names
    :return features: (string array) column headers for the features (without the label header)
    :return data: (numpy array) feature values as in the given dataset, without the labels.
    """
    # Prepare the dataset for clustering algorithms
    features = dataset.columns.values # Get the features names (based on headers)
    features = np.delete(features, -1) # Delete the last element (column 'sample') NOTE: THIS IS BECAUSE IN MY DATASETS, THE LABEL COLUMN IS ALWAYS THE LAST ONE, CHANGE ACCORDINGLY
    data = dataset.loc[:, features].values # Separating out the features
    target = dataset.loc[:, labelHeader].values # Separating the label column (sample type)

    if minmax: # In case we want to normalize by using min max scaler (for the MST)
        scaledX = MinMaxScaler().fit_transform(data) #Scale data to a 0-1 range
        targetMass = None # We don't use this in the mst algorithm
    else: #We use standardization (pca, umap and t-sne)
        # Add mean to the dataset --> I plot it together with the clustering algorithms
        massCenters = pd.DataFrame.from_dict(mean, orient='index') #Convert dictionary into dataframe
        targetMass = np.append(target, massCenters.index) #Append massCenter's sample names to target array
        data = np.append(data, massCenters.to_numpy(), axis=0) #Append massCenter's sample
        scaledX = StandardScaler().fit_transform(data) # Standardizing the features (z-score)

    # Find unique sample names
    uniqueLabels = list(set(target)) # Get unique labels
    uniqueLabels = sorted(uniqueLabels) # Sort them alphabetically 

    return scaledX, target, targetMass, uniqueLabels, features, data


##################################################################
###################### CLUSTERING FUNCTIONS ######################
##################################################################

def pca2D(features, target, scaledX, uniqueLabels, colorDict, nFeatures):

    pcaTotal = PCA(random_state=2023) # set a random seed, we don't specify number of components to see how is the 100% of the variance explained (PCA with all features)
    principalComponents = pcaTotal.fit_transform(scaledX) # Apply pca to the standardized dataset
    print("Explained variance by all components: ", sum(pcaTotal.explained_variance_ratio_*100))

    pca_df = pd.DataFrame({'pca_1': principalComponents[:,0], 'pca_2': principalComponents[:,1], 'label': target}) # Get the 2 PC for the scatter plot

    # Calculate the feature importance based on the loadings
    featureImportanceTotal = pd.DataFrame(data=pcaTotal.components_, columns=features)
    print(featureImportanceTotal.abs().idxmax(axis=1).head(nFeatures))
    featureImportanceTotal.to_csv("pca_featureImportance.csv") # Save feature importance in a csv
    topF = featureImportanceTotal.abs().idxmax(axis=1).head(nFeatures) # Save top 10 features
    loadingsDF = pd.DataFrame({'x':featureImportanceTotal[topF].iloc[0,:], 'y':featureImportanceTotal[topF].iloc[1,:]}) # Create dataframe with the loadings for the simplicity of plotting
    loadingsDF.to_csv("pca_top10_PC1-PC2.csv") # Save top 10 loadings in csv

    # Plot elbow plot
    sns.set_theme(style="darkgrid")
    figElbow, ax = plt.subplots()
    ax.set(xticks=np.arange(features.shape[0]))
    ax.plot(np.cumsum(pcaTotal.explained_variance_ratio_)) # Elbow plot: explained variance by each component
    plt.xlabel('Number of components')
    plt.ylabel('Explained variance (%)')
    figElbow.suptitle("Elbow plot with explained variance")
    figElbow.show()
    figElbow.savefig('Elbow_plot.svg', transparent=True)
    figElbow.savefig('Elbow_plot.png', transparent=True, dpi=600)

    # Plot biplot
    pcaBiplot(pcaTotal, pca_df, loadingsDF, nFeatures, colorDict)

    # Classic PCA plot
    explainedVariance = pcaTotal.explained_variance_ratio_
    samplePlots = True
    plot(pca_df, 'PCA', 'pca_1', 'pca_2', uniqueLabels, colorDict, samplePlots, explainedVariance)


def tsneTuner(scaledX, target, perplexities, colorDict):
    # Hypertune t-SNE
    counter = 1 #for the subplots
    fig = plt.figure(figsize=(8,8))
    fig.suptitle("Perplexity hypertuning t-SNE")

    for p in perplexities:
        tsne = TSNE(n_components=2, perplexity=p, random_state=23)
        tsne_result = tsne.fit_transform(scaledX)

        subFig = fig.add_subplot(3,3,counter)
        subFig.set_title("Perplexity = " + str(p) + "\nKL divergence = " + str("%.2f" % round(tsne.kl_divergence_,2)))
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': target}) #Prepare data into a dataframe to plot
        sns.set_theme(style="darkgrid")
        clustFullData = tsne_result_df[tsne_result_df["label"].str.contains("massCenter")!=True] #Remove the mass centers from this plot
        sns.scatterplot(x='tsne_1', y='tsne_2', data=clustFullData, hue='label', palette=colorDict, alpha=0.7, legend=False)

        print("\n\nPerplexity = ", str(p))
        print("Estimator parameters = ", tsne.get_params())
        print("Kullback-Leibler divergence = ", tsne.kl_divergence_)
        print("Effective learning rate = ", tsne.learning_rate_)
        counter += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc='upper right') #plot legend
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.tight_layout()
    plt.show()

    ## Save figure ##
    Path('TSNE').mkdir(parents=True, exist_ok=True)
    fig.savefig("TSNE/perplexityTest.svg", facecolor=(1,1,1,0), dpi=72, bbox_inches='tight')
    fig.savefig("TSNE/perplexityTest.png", facecolor=(1,1,1,0), dpi=600, bbox_inches='tight')


def mytsne(data, target, perplexity, uniqueLabels, colorDict):

    # We want to get TSNE embedding with 2 dimensions
    tsne = TSNE(n_components=2, perplexity=perplexity, verbose=1, random_state=42)
    tsne_result = tsne.fit_transform(data)
    print("Estimator parameters = ", tsne.get_params())
    print("Kullback-Leibler divergence = ", tsne.kl_divergence_)
    print("Effective learning rate = ", tsne.learning_rate_)

    # Plot the result of our TSNE with the label color coded
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': target})

    plot(tsne_result_df, 'TSNE', 'tsne_1', 'tsne_2', uniqueLabels, colorDict)


def umapTuner(scaledX, target, n_neighbors, colorDict):
    # Hypertune UMAP
    counter = 1 #for the subplots
    fig = plt.figure(figsize=(8,8))
    fig.suptitle("Number of neighbors hypertuning UMAP")

    for n in n_neighbors:
        reducer = UMAP.UMAP(n_neighbors=n, min_dist=0.1, random_state=23)
        umap_result = reducer.fit_transform(scaledX)

        subFig = fig.add_subplot(3,3,counter)
        subFig.set_title("n_neighbors = " + str(n))
        umap_result_df = pd.DataFrame({'umap_1': umap_result[:,0], 'umap_2': umap_result[:,1], 'label': target}) #Prepare data into a dataframe to plot
        sns.set_theme(style="darkgrid")
        clustFullData = umap_result_df[umap_result_df["label"].str.contains("massCenter")!=True] #Remove the mass centers from this plot
        sns.scatterplot(x='umap_1', y='umap_2', data=clustFullData, hue='label', palette=colorDict, alpha=0.7, legend=False)
        counter += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc='upper right') #plot legend
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.tight_layout()
    plt.show()
    ## Save figure ##
    Path('UMAP').mkdir(parents=True, exist_ok=True)
    fig.savefig("UMAP/n_neighborsTest.svg", facecolor=(1,1,1,0), dpi=72, bbox_inches='tight')
    fig.savefig("UMAP/n_neighborsTest.png", facecolor=(1,1,1,0), dpi=600, bbox_inches='tight')


def myumap(scaledX, target, n_neighbors, uniqueLabels, colorDict, samplePlots):

    reducer = UMAP.UMAP(n_neighbors=n_neighbors, random_state=25, verbose=True) # Get umap object
    embedding = reducer.fit_transform(scaledX) # Train the reducer
    embedding_df = pd.DataFrame({'umap_1': embedding[:,0], 'umap_2': embedding[:,1], 'label': target}) # Prepare dataframe for plotting

    plot(embedding_df, 'UMAP', 'umap_1', 'umap_2', uniqueLabels, colorDict, samplePlots)
    return reducer, embedding_df


def featureColoredUmap(dataset, featureToColor, clustData, x, y):
    clustFullData1 = clustData[clustData["label"].str.contains("massCenter")!=True] #Remove mean class
    clustFullData = pd.concat([clustFullData1.reset_index(drop=True), dataset[featureToColor].reset_index(drop=True)], axis=1) #Concat features to embeddings

    ## SMALL PATCH TO RENAME THE FEATURES FOR THE PAPER FIGURES ##
    clustFullData.rename(columns={'channel 1 target count':'Fab',	'channel 2 target count':'Fc'}, inplace=True)
    featureToColor = ['Diameter', 'Aspect ratio', 'Cluster localizations', 'Fab', 'Fc', 'channel 1 true mean dark time', 'channel 2 true mean dark time', 'channel 1 mean bright time', 'channel 2 mean bright time']
    ###

    for feature in featureToColor:
        fig, ax = plt.subplots(1, figsize=(15,15))
        sns.set_theme(style="darkgrid")
        sns.scatterplot(x=x, y=y, data=clustFullData, hue=feature, palette='crest', ax=ax, alpha=0.7, s=150)
        plt.gca().set_aspect('equal', 'datalim')
        plt.xlabel("UMAP 1", fontsize=34)
        plt.ylabel("UMAP 2", fontsize=34)
        plt.yticks([])
        plt.xticks([])
        plt.title(feature, fontsize=42)

        plt.legend(fontsize=32, loc='upper right') #plot legend

        fig.tight_layout()
        plt.show(block=False)
        # fig.set_size_inches((11, 11), forward=False)
        Path('UMAP/featureColored').mkdir(parents=True, exist_ok=True)
        fig.savefig('UMAP/featureColored/UMAP'+'_'+feature+".svg", facecolor=(1,1,1,0), dpi=72)
        fig.savefig('UMAP/featureColored/UMAP'+'_'+feature+".png", facecolor=(1,1,1,0), dpi=600)


def mymst(data, target, colorDict, subFraction, filename):
    # minimum spanning trees clustering --> connects the nonzero minimum weight nodes of a given graph.

    ## Transform dataset into a graph:
    p = data.shape[1] # Get number of features
    D = pairwise_distances(data) # Calculate euclidean distance
    print("Calculated pairwise Euclidean")
    D = D/np.sqrt(p) # Calculate average eucliden distance (for each element)
    triD = np.tril(D) # Get the lower triangular matrix, that is to not repeat distances
    print("Lower triangular matrix done!")

    # Create a mst from our graph:
    triD = triD.astype(np.float32) # transform to float32 to save memory space
    gc.collect() # garbage collector (to free up memory space)
    print("Calculating minimum spanning tree...")
    tree = minimum_spanning_tree(triD) # Create tree
    treeArray = tree.toarray().astype(np.float32)
    tree = None # Free up memory

    # Organized nodes (pre-calculating node position):
    print("calculating Kamada Kawai layout")
    pos = nx.kamada_kawai_layout(nx.Graph(treeArray)) #Calculate node positions

    # Plot the tree:
    colorArray = [colorDict[sample] for sample in target] #Get a color array for the target (Translates target to the HEX color in our colormap dictionary) --> nodes are in the same order as our class
    edgeColorArray = [colorDict[target[edge[1]]] for edge in list(nx.Graph(treeArray).edges())] # Get the color corresponding to the target node of the edge (n2). We index the target vector with the target node id: n1 --> n2 == edge (-->) will be the same color as node n2. To color based on the source node you can use target[edge[0]]
    gc.collect() #garbage collector (to free up memory space)

    figNodes, ax = plt.subplots(1, figsize=(15,15))
    sns.set_theme(style="darkgrid")
    nx.draw(nx.Graph(treeArray), pos=pos, node_shape='.', with_labels=False, node_color=colorArray, node_size=60, alpha=0.6, edge_color=edgeColorArray) #NODES
    plt.draw()

    plt.show(block=False)

     ## Save figure ##
    Path('MST').mkdir(parents=True, exist_ok=True)
    figNodes.savefig("MST/MST_nodes_frac"+str(subFraction)+str(filename)+".svg", facecolor=(1,1,1,0), dpi=72, bbox_inches='tight')
    figNodes.savefig("MST/MST_nodes_frac"+str(subFraction)+str(filename)+".png", facecolor=(1,1,1,0), dpi=600, bbox_inches='tight')

    ############ Repeat plot but with sticks ################
    figSticks, ax = plt.subplots(1, figsize=(15,15))
    sns.set_theme(style="darkgrid")

    nx.draw(nx.Graph(treeArray), pos=pos, with_labels=False, nodelist=[], edge_color=edgeColorArray, width=1) #STICKS
    plt.draw()

    plt.show(block=False)

    Path('MST').mkdir(parents=True, exist_ok=True)
    figSticks.savefig("MST/MST_frac"+str(subFraction)+str(filename)+".svg", bbox_inches='tight', facecolor=(1,1,1,0), dpi=72)
    figSticks.savefig("MST/MST_frac"+str(subFraction)+str(filename)+".png", bbox_inches='tight', facecolor=(1,1,1,0), dpi=600)


def silhouetteCoefficient(data, target, uniqueLabels, colorDict, labelHeader, pairwise, figsize):
    ### FULL OVERVIEW ###
    silhouetteScores = silhouette_samples(data, target) # Use the default Euclidean distance to calculate the Silhouette score (same as the pre-computed one)
    silhouetteAverage = silhouette_score(data, target) # Compute the average Silhouette score, to have an idea of the density of our plot.
    print("Average Silhouette score: ", silhouetteAverage)

    # Create long dataframe with the scores per sample (this way the violin plot can show the imbalance in the dataset)
    silhouetteDF = pd.DataFrame() # Init dataframes
    silhouetteTmp = pd.DataFrame()
    for sample in uniqueLabels:
        silhouetteTmp['Score'] = silhouetteScores[target == sample] # Get score per sample
        silhouetteTmp[labelHeader] = sample # Add column with label

        silhouetteDF = pd.concat([silhouetteDF, silhouetteTmp], ignore_index=True, axis=0) # Add new sample scores to the dataframe
        silhouetteTmp = pd.DataFrame() # Reset temporal variable after use

    fig, ax = plt.subplots(1, figsize=figsize) 
    sns.set_theme(style="darkgrid")
    ax.set_ylim([-1, 1]) # Set limits to Silhouette Score range
    if pairwise:
        sns.violinplot(data=silhouetteDF, x=labelHeader, y='Score', cut=0, split=True, gap=.1, palette=colorDict, saturation=1, inner=None, common_norm=True, density_norm='count')
    else:
        sns.violinplot(data=silhouetteDF, x=labelHeader, y='Score', palette=colorDict, hue=labelHeader, legend=False, saturation=1, density_norm='count', common_norm=True, inner='quart',  cut=0)
    ax.text(-0.5, 0.75, "Grouped", rotation = 'vertical')
    ax.text(-0.5, -0.15, "Overlapping", rotation = 'vertical')
    ax.text(-0.5, -1, "Misgrouped", rotation = 'vertical')
    if pairwise:
        ax.set_title("SILHOUETTE SCORES\npairwise comparison\n", fontsize=20)
    else:
        ax.set_title("SILHOUETTE SCORES\ndataset overview", fontsize=20)
    ax.axhline(y=silhouetteAverage, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="grey", alpha=0.5)
    if pairwise:
        plt.xticks([])
        ax.set_xlabel('')
    else:
        plt.xticks(rotation=45, ha='right')
    plt.show(block=False)

    ## Save figure ##
    Path('Silhouette').mkdir(parents=True, exist_ok=True)
    if pairwise:
        fig.savefig("Silhouette/silhouetteScores_"+uniqueLabels[0]+"_"+uniqueLabels[1]+".svg", facecolor=(1,1,1,0), dpi=72, bbox_inches='tight')
        fig.savefig("Silhouette/silhouetteScores_"+uniqueLabels[0]+"_"+uniqueLabels[1]+".png", facecolor=(1,1,1,0), dpi=600, bbox_inches='tight')
    else:
        fig.savefig("Silhouette/silhouetteScores.svg", facecolor=(1,1,1,0), dpi=72, bbox_inches='tight')
        fig.savefig("Silhouette/silhouetteScores.png", facecolor=(1,1,1,0), dpi=600, bbox_inches='tight')
    

def metrics(dataset, labelHeader):
    mcvArray = [] # Initialize array to save mcv for each sample
    meanArray = [] # Initialize array to save mass center for each sample
    pdiArray = [] # Initialize array to save PDI for each sample
   
    #calculate for each sample:
    uniqueNameSamples = list(set(dataset[labelHeader]))
    for sampleName in uniqueNameSamples:
        data = dataset[dataset[labelHeader]==sampleName] # Save only elements from one sample
        data.drop(columns=[labelHeader], inplace=True) # Drop non-numeric column
        
        # Function for multivariate CVs 
        n = len(data) #number of samples (rows)
        p = len(data.columns) #number of features (columns)

        # Classical estimates
        mean = data.mean(axis=0) # Mean (column wise)
        s = data.cov() # Covariance matrix
        sd = data.std() # get Standard deviation
        CV = sd.values/mean.values # Get CV for each feature
        # Calculate the PDI
        pdi = np.square(CV)


        # MCV (with full dataset)
        MCV = np.sqrt(np.dot(np.dot(mean.values, s.values), mean.values.T))/(np.dot(mean.values, mean.values.T)**2)*100 #np.dot is the only way of transposing vectors with numpy
        try:
            MCVvn = ((np.dot(np.dot(mean.values,np.linalg.inv(s.values)),mean.values.T))**-0.5)*100
        except np.linalg.LinAlgError:
            MCVvn = -1
            print('MCVvn cannot be calculated because the covariance matrix is a singular matrix')
        
        mcvArray.append(MCVvn)
        meanArray.append(mean)
        pdiArray.append(pdi)

        # Print results
        print("\nSample name: ", sampleName)
        print("sampleSize: ", n)
        print("Center of mass: ", mean.values)
        print("Standard deviations: ", sd.values)
        print("\nCov matrix:\n", s)
        print('MCV(%) = '+str("%.5f" % round(MCV,5))+'\nMCVvn(%) = '+str("%.2f" % round(MCVvn,2)))


    mcvDict = {uniqueNameSamples[i]: mcvArray[i] for i in range(len(uniqueNameSamples))} #Create a dictionary with the mcv for each sample
    meanDict = {uniqueNameSamples[i]+'_massCenter': meanArray[i] for i in range(len(uniqueNameSamples))} #Create a dictionary with the mass center for each sample
    pdiDict = {uniqueNameSamples[i]: pdiArray[i] for i in range(len(uniqueNameSamples))} #Create a dictionary with the pdi for each sample

    return s.index, pdiDict, mcvDict, meanDict


##################################################################
####################### PLOTTING FUNCTIONS #######################
##################################################################

def metricsPlot(smaller_dataset, ticks, MCVvn, samples, compare, labelHeader):
    if compare: #Find higher CV value to make axis equal
        ylim = 0 # Initialize axis limit
        for sampleName in samples:
            smallData = smaller_dataset[smaller_dataset[labelHeader]==sampleName] # Same but for the smaller dataset
            smallData.drop(columns=[labelHeader], inplace=True) # Same but for the smaller dataset
            smallCV = smallData.std().values/smallData.mean(axis=0).values*100 #Get CV for only the smaller dataset
            if smallCV.max() > ylim: 
                ylim = smallCV.max() # Update limit

    for sampleName in samples:
        smallData = smaller_dataset[smaller_dataset[labelHeader]==sampleName] # Same but for the smaller dataset
        smallData.drop(columns=[labelHeader], inplace=True) # Same but for the smaller dataset
        smallCV = smallData.std().values/smallData.mean(axis=0).values*100 #Get CV for only the smaller dataset

        # Plotting
        # Edit seismic colormap to have a broader white range ##
        p = [-1, -0.25, 0.25, 1] # Coordinates of the data points (our color boundaries)
        f = lambda x: np.interp(x, p, [0, 0.5, 0.5, 1]) # Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (p, y-coords), evaluated at x.

        cmap = LinearSegmentedColormap.from_list('map_white', 
                    list(zip(np.linspace(0,1), plt.cm.seismic_r(f(np.linspace(min(p), max(p))))))) # Get the reversed seismic colormap, with our personalized boundaries defined in function f

        meansAnnot = np.round(smallData.mean(axis=0).values, 1)
        ratio = len(ticks)
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, ratio], 'width_ratios': [ratio, 0.2], 'hspace':0.1, 'wspace':0.01})
        plt.rcParams['font.family'] = 'DejaVu Sans'
        fig.suptitle('Metrics Matrix\n'+sampleName+'\n', fontsize=24)

        # Plot small barplot with the CVs for each of the selected features
        ax[0,0].bar(ticks, smallCV, color = 'lightgrey', tick_label='')
        if compare:
            ax[0,0].set_ylim(0,ylim) # Make axis equal when making a comparison
        ax[0,0].tick_params(bottom=False)
        ax[0,0].tick_params(axis='y', labelsize = 12) # Delete ticks marks push the x labels down
        ax[0,0].set_ylabel('CV (%)', fontsize=16)
        ax[0,0].set_title("\nMCV = "+str("%.2f" % round(MCVvn[sampleName],2))+"%", fontsize=20)
        ax[0,0].set_position([0.19,0.75, 0.61, 0.1])
        sns.despine() # Delete axis borders
        # Hide extra axis used for alignment
        ax[0,1].spines['left'].set_visible(False) # Remove border from axis in the big heatmap
        ax[0,1].spines['bottom'].set_visible(False) # Remove border from axis in the big heatmap
        ax[0,1].tick_params(axis='x', bottom=False, labelbottom=False) # Delete ticks marks push the x labels down
        ax[0,1].tick_params(axis='y', left=False, labelleft=False) # Delete ticks marks push the x labels down

        # Add a table with the feature means at the bottom of the figure:
        table = ax[1,0].table(cellText=np.round(meansAnnot.reshape(1,len(smallCV)),2), rowLabels=['Mean:'], rowLoc='center', cellLoc='center', loc='bottom')
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(1, 1.3)

        # Plot main heatmap with the correlation coefficients for each pairwise of the selected features
        sns.heatmap(smallData.corr(), ax=ax[1,0], linewidths = .3, linecolor='whitesmoke', yticklabels=ticks, xticklabels=ticks, cmap=cmap, vmin=-1, vmax=1, square=True, cbar_ax=ax[1,1], cbar_kws={'label': 'Pearson\'s correlation coefficient'}) # Plot correlation matrix --  -- cbar_kws={'fontsize': 14, "aspect": 20/w, 'label': 'Pearson\'s correlation coefficient'}
        # Fix cbar ticks
        ax[1,1].tick_params(axis='x', pad=11, labelsize = 12, bottom=False) # Delete ticks marks push the x labels down
        ax[1,1].tick_params(axis='y', labelsize = 12, left=False) # Delete ticks marks push the x labels down
        ax[1,1].spines['left'].set_visible(False) # Remove border from axis in the big heatmap
        # Fix heatmap ticks
        ax[1,0].tick_params(axis='x', pad=15, bottom=False, labelsize=20, labelrotation=0) # Delete ticks marks push the x labels down
        ax[1,0].tick_params(axis='y', left=False, labelsize=20, labelrotation=360) # Delete ticks marks push the x labels down
        ax[1,0].spines['left'].set_visible(False) # Remove border from axis in the big heatmap

        plt.grid(False)
        plt.show(block=False)
        
        # Save figure
        Path('metricsMatrix').mkdir(parents=True, exist_ok=True)
        if compare:
            sampleName = sampleName+'_compare'
        fig.savefig("metricsMatrix/"+sampleName+"_metricsMatrix.svg", bbox_inches='tight', transparent=True, dpi=72)
        fig.savefig("metricsMatrix/"+sampleName+"_metricsMatrix.png", bbox_inches='tight', transparent=True, dpi=600)

        # Save csv to get exact values for the heatmap
        smallData.corr().to_csv("metricsMatrix/"+sampleName+"_metricsMatrix.csv")
        # Same but for CV values
        pd.DataFrame(smallCV, index=ticks).to_csv("metricsMatrix/"+sampleName+"_CV.csv")
       

def plot(clustData, clustMethod, x, y, uniqueLabels, colorDict, samplePlots=True, explainedVariance=None): #Function to plot the results from the clustering algorithms
    #Drop means to plot in full color
    clustFullData = clustData[clustData["label"].str.contains("massCenter")!=True]

    #Plot all samples in full color
    if clustMethod!='PCA':
        fig, ax = plt.subplots(1, figsize=(15,15))
        sns.set_theme(style="darkgrid")
        sns.scatterplot(x=x, y=y, data=clustFullData, hue='label', palette=colorDict, ax=ax, alpha=0.7, s=200, edgecolor='#B3B3B3', linewidth=0.5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.xlabel(clustMethod+" 1", fontsize=34)
        plt.ylabel(clustMethod+" 2", fontsize=34)
        plt.yticks([])
        plt.xticks([])
        plt.title(clustMethod, fontsize=42)

        plt.legend('', frameon=False)

        fig.tight_layout()
        plt.show(block=False)
        # fig.set_size_inches((11, 11), forward=False)
        Path(clustMethod).mkdir(parents=True, exist_ok=True)
        if samplePlots:
            strVar = '_complete'
        else:
            strVar = '_pairwise_'+uniqueLabels[0]+'_'+uniqueLabels[1]
        fig.savefig(clustMethod+"/"+clustMethod+strVar+".svg", facecolor=(1,1,1,0), dpi=72)
        fig.savefig(clustMethod+"/"+clustMethod+strVar+".png", facecolor=(1,1,1,0), dpi=600)
    
    if samplePlots:
        for label in uniqueLabels:
            #Find corresponding mass center:
            fign, axn = plt.subplots(1, figsize=(15,15))
            plt.gca().set_aspect('equal', 'datalim')
            sns.set_theme(style="darkgrid")
            sns.scatterplot(x=x, y=y, data=clustData[clustData['label'] != label], linewidth=0.5, s=200, color="none", edgecolor='#A7A7A7', alpha=0.7, marker='o') #Grey for background
            sns.scatterplot(x=x, y=y, data=clustData[clustData['label'] == label], linewidth=2, s=275, color="none", edgecolor=colorDict[label], alpha=1, marker='o') #Color based on the palette
            sns.scatterplot(x=x, y=y, data=clustData[clustData['label'] == label+'_massCenter'], linewidth=1.75, s=275, color='#D46C6C', edgecolor='#FFFFFF', alpha=1, marker='o') #Plot mass center for the corresponding sample
            plt.gca().set_aspect('equal', 'datalim')
            plt.xlabel(clustMethod+" 1", fontsize=34)
            plt.ylabel(clustMethod+" 2", fontsize=34)
            plt.title(clustMethod+' for sample '+label, fontsize=42)

            if clustMethod=='PCA':
                axn.axhline(0, linestyle='--', color='k', alpha=0.4) # horizontal lines
                axn.axvline(0, linestyle='--', color='k', alpha=0.4) # vertical lines
                axn.set_xlim(-15,15)
                axn.set_ylim(-15,15)
                plt.yticks(fontsize=32)
                plt.xticks(fontsize=32)
                axn.set_xlabel('Scores on PC1\nExplained variance = '+str("%.2f" % round(explainedVariance[0]*100,2)+"%"), fontsize=34)
                axn.set_ylabel('Scores on PC2\nExplained variance = '+str("%.2f" % round(explainedVariance[1]*100,2)+"%"), fontsize=34)
            else:
                plt.yticks([])
                plt.xticks([])

            fign.tight_layout()
            plt.show(block=False)
            # fign.set_size_inches((11, 11), forward=False)
            figTitle = clustMethod+'/'+clustMethod+'_'+label
            fign.savefig(figTitle+'.svg', facecolor=(1,1,1,0), dpi=72)
            fign.savefig(figTitle+'.png', facecolor=(1,1,1,0), dpi=600)


def pcaBiplot(pcaTotal, pca_df, loadingsDF, nFeatures, colorDict):
    fig, ax = plt.subplots(1, figsize=(15,15))
    sns.set_theme(style="darkgrid", font_scale=2)
    plt.title('PCA biplot', fontsize=42)

    # Plot scores
    #Drop means to plot in full color
    clustFullData = pca_df[pca_df["label"].str.contains("massCenter")!=True]

    f = sns.scatterplot(x='pca_1', y='pca_2', data=clustFullData, hue='label', palette=colorDict, ax=ax, alpha=0.65, s=200, edgecolor='#B3B3B3')
    ax.axhline(0, linestyle='--', color='k', alpha=0.4) # horizontal lines
    ax.axvline(0, linestyle='--', color='k', alpha=0.4) # vertical lines
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    plt.gca().set_aspect('equal', 'datalim')
    plt.xlabel('Scores on PC1\nExplained variance = '+str(np.round(pcaTotal.explained_variance_ratio_[0]*100,2))+'%', fontsize=34)
    plt.ylabel('Scores on PC2\nExplained variance = '+str(np.round(pcaTotal.explained_variance_ratio_[1]*100,2))+'%', fontsize=34)
    plt.yticks(fontsize=32)
    plt.xticks(fontsize=32)

    # Plot loadings
    augm = 45 #Loadings are multiplied by this number so they're visible in the plot
    [plt.arrow(0,0,loadingsDF.iloc[i,0]*augm, loadingsDF.iloc[i,1]*augm, color='r', alpha=1, length_includes_head=True, head_width=0.5) for i in range(nFeatures)] # Multiply by 200 so the arrows are visible in the scores scale
    texts = [plt.text(loadingsDF.iloc[line,0]*augm, loadingsDF.iloc[line,1]*augm, loadingsDF.index[line], horizontalalignment='left', size='medium', color='r') for line in range(nFeatures)] # Add annotation to the loadings
    adjust_text(texts) # add arrows to the plot: , arrowprops=dict(arrowstyle='->', color='red'), , only_move='y'

    plt.legend('', frameon=False) # Remoce legend
    
    ## Save figure ##
    Path('PCA').mkdir(parents=True, exist_ok=True)
    fig.savefig("PCA/biplot.svg", facecolor=(1,1,1,0), dpi=72, bbox_inches='tight')
    fig.savefig("PCA/biplot.png", facecolor=(1,1,1,0), dpi=600, bbox_inches='tight')


def plotLegend(colorDict):
        ### ATLAS' forced legend
        fig, ax = plt.subplots(1, figsize=(5,2))
        sns.set_theme(style="darkgrid")
        # Remove ticks
        plt.yticks([])
        plt.xticks([])
        # create patch
        values = list(colorDict.values())
        keys = list(colorDict.keys())
        patch = [(mpatches.Patch(color=values[i], label=keys[i])) for i in range(len(colorDict))] #Markers are rectangles
        plt.legend(handles = patch, fontsize=32, title='Samples', title_fontsize=32, loc='upper right') #plot legend
        fig.savefig('legend.svg', facecolor=(1,1,1,0), dpi=72)


def scatterEDA(dataset, x, y1, y2, title, colorDict, labelHeader, filename):
    fig, ax = plt.subplots(2, figsize=(15,15))

    ## Channel 1
    sns.scatterplot(dataset, ax = ax[0], x=x, y=y1, hue=labelHeader, palette=colorDict)
    ax[0].set_xlim(0,1400)
    ax[0].set_ylim(0,1400)

    ax[0].axline((1, 1), slope=1, color='grey', ls='--') # Add diagonal (x=y)

    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    ax[0].legend(fontsize=14, loc='lower right')
    ax[0].set_xlabel('Total cluster localizations', fontsize=16)
    ax[0].set_ylabel(y1, fontsize=16)

    ## Channel 2
    sns.scatterplot(dataset, ax = ax[1], x=x, y=y2, hue=labelHeader, palette=colorDict)
    ax[1].set_xlim(0,1400)
    ax[1].set_ylim(0,1400)

    ax[1].axline((1, 1), slope=1, color='grey', ls='--') # Add diagonal (x=y)

    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].legend(fontsize=14, loc='lower right')
    ax[1].set_xlabel('Total cluster localizations', fontsize=16)
    ax[1].set_ylabel(y2, fontsize=16)

    fig.suptitle(title, fontsize=20, y=0.92)

    ## Save figure ##
    Path('EDA').mkdir(parents=True, exist_ok=True)
    fig.savefig("EDA/scatterEDA"+filename+".png", facecolor=(1,1,1,0), dpi=600, bbox_inches='tight')
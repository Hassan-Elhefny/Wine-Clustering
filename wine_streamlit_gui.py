# Import Libraries
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Read wine Dataset
wine_df = pd.read_csv('wine-clustering.csv')
standard_scaler = StandardScaler()
wine_df_scaled = standard_scaler.fit_transform(wine_df)
wine_df_scaled_df = pd.DataFrame(wine_df_scaled, columns=wine_df.columns)


def plot_data_in_two_d(dataset, title):
    pca = PCA(n_components=2, random_state=42)
    two_d = pca.fit_transform(dataset)
    fig = px.scatter(
        x=two_d[:, 0],
        y=two_d[:, 1],
        title=title
    ).update_layout(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2'
    )
    return fig


def plot_data_in_three_d(dataset, title):
    pca = PCA(n_components=3, random_state=42)
    three_d = pca.fit_transform(dataset)
    fig = px.scatter_3d(
        x=three_d[:, 0],
        y=three_d[:, 1],
        z=three_d[:, 2],
        title=title
    )
    return fig


inertia_scores = []
silhouette_scores = []
for i in range(2, 13):
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(wine_df_scaled_df)
    inertia_scores.append(model.inertia_)
    silhouette_scores.append(silhouette_score(
        wine_df_scaled_df, model.labels_))

# *************************************************************************************************
# Set Initial Page Configration
st.set_page_config(
    page_title='Wine Clustering',
    layout="wide",
    initial_sidebar_state="expanded"
)

st.set_option('deprecation.showPyplotGlobalUse', False)


# *************************************************************************************************
# Title
st.header(
    "Wine Clustering Project"
)

st.sidebar.header('Clustering Kaggle Competition')
st.sidebar.image(
    "https://www.gannett-cdn.com/-mm-/a481916a9da4934da6f90fc763e8898a9ede1f3d/c=0-248-4928-3032/local/-/media/2015/05/27/KUSA/KUSA/635683134798795203-GettyImages-456025746.jpg?width=3200&height=1680&fit=crop",
    width=300
)
st.sidebar.write(
    """
    ### I Seperate This Project to three main parts:
    ### 1. Description: in this part i describe and plot the data.
    ### 2. K-Means: in this part i discuss what is k-means and how this algorithm work and also focus on three different mitrics to get the best value of k.
    ### 3. DBSCAN: in this part i discuss what is DBSCAN and how this algorithm work.
    """
)


main_parts = ['Description', 'KMeans', 'DBSCAN']

st.sidebar.header("")
user_request = st.sidebar.radio(
    'Select What You Want to Do : ',
    main_parts
)

if user_request == main_parts[0]:
    # Project Description
    st.write("""
    #### In This Project we Will Use UnSupervised learning techniques to group the similar types of wine together.  

    ## Clustering Techniques:
    ### 1. k-Means
    ### 2. DBBSCAN

    ##### This dataset is adapted from the Wine Data Set from https://archive.ics.uci.edu/ml/datasets/wine by removing the information about the types of wine for unsupervised learning.

    ##### The following descriptions are adapted from the UCI webpage:

    ##### These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

    ## The attributes are:
    ### 1. Alcohol
    ### 2. Malic acid
    ### 3. Ash
    ### 4. Alcalinity of ash
    ### 5. Magnesium
    ### 6. Total phenols
    ### 7. Flavanoids
    ### 8. Nonflavanoid phenols
    ### 9. Proanthocyanins
    ### 10. Color intensity
    ### 11. Hue
    ### 12. OD280/OD315 of diluted wines
    ### 13. Proline

    """)
    # Image
    st.image(
        'https://media.winefolly.com/whole-cluster-fermentation-destemmed-illustration-winefolly.png',
        caption='Wine Image',
        width=700
    )
    st.header("Wine Dataset")
    st.write(wine_df)
    st.header("Plot Data In 2D")
    fig = plot_data_in_two_d(wine_df, 'Wine Dataset in 2D')
    st.write(fig)
    st.header("Plot Data In 3D")
    fig = plot_data_in_three_d(wine_df, 'Wine Dataset in 3D')
    st.write(fig)
    # Information about wine dataset
    st.header(
        "Min, Max, Std and Mean before Scale The data"
    )
    st.write(
        pd.DataFrame(wine_df.aggregate(
            ['min', 'mean', 'std', 'max']).to_dict())
    )
    # Information about wine dataset
    st.header(
        "Min, Max, Std and Mean After Scale The data using Standard Scaler"
    )
    st.write(
        pd.DataFrame(wine_df_scaled_df.aggregate(
            ['min', 'mean', 'std', 'max']).to_dict())
    )
elif user_request == main_parts[1]:
    st.header('K-Means Clustering')
    st.write(
        """
        K-means clustering is a popular unsupervised machine learning algorithm used for clustering data.
        The goal of k-means clustering is to partition a given dataset into k clusters, where k is a predefined number. 
        The algorithm works by iteratively assigning each data point to the nearest centroid (center) of the cluster, 
        and then recalculating the centroids based on the newly formed clusters. The algorithm stops when the centroids 
        no longer move significantly or a predefined number of iterations is reached.

        ### Here are the main steps of the k-means clustering algorithm:
        1. Initialize k centroids randomly in the feature space.
        2. Assign each data point to the nearest centroid.
        3. Recalculate the centroids based on the newly formed clusters.
        4. Repeat steps 2-3 until the centroids no longer move significantly or a predefined number of iterations is reached.

        K-means clustering has several advantages, including its simplicity and scalability to large datasets. However, 
        it also has some limitations, such as its sensitivity to the initial choice of centroids and its tendency to converge to local minima. Therefore, 
        it is often recommended to run the algorithm multiple times with different initial centroids to obtain more robust results.
        """
    )
    st.header("Choose The Best Value Of K using three different metrics")
    st.header('1. Distortion')
    st.write(
        """
        In the context of clustering, distortion refers to the average distance between the data points and their assigned centroid in a given clustering solution. 
        Distortion is also known as the clustering error or the within-cluster sum of squares (WCSS). 
        The objective of k-means clustering is to minimize the distortion, which implies that we want to find the clustering solution that leads to the smallest average distance between data points and their assigned centroids.

        ### The distortion can be calculated as follows:

        1. For each cluster, calculate the sum of the squared distances between the data points and their assigned centroid.
        2. Sum the results from step 1 over all clusters.
        3. Divide the result from step 2 by the total number of data points to obtain the average distortion.

        The k-means algorithm tries to minimize the distortion by iteratively re-assigning data points to their nearest centroid and recalculating the centroids until convergence.
        One limitation of using distortion as a measure of clustering quality is that it tends to decrease as the number of clusters increases, regardless of whether the additional clusters actually represent meaningful partitions of the data. Therefore, it is often recommended to use other measures, such as the silhouette score or the gap statistic, to determine the optimal number of clusters.
        """
    )
    st.image(
        "https://www.codeproject.com/KB/AI/5256294/Fig6.png",
        width=600,
        caption="Distortion"
    )
    kmeans_model = KMeans(random_state=42)
    visualizer = KElbowVisualizer(kmeans_model, k=(2, 12))
    visualizer.fit(wine_df_scaled_df)
    st.header("Elbow Method to choose The Best Value Of K")
    visualizer.show()
    st.pyplot()
    st.header('2. Inertia')
    st.write(
        """
        Inertia is a metric used to evaluate the quality of clustering solutions, particularly in the context of K-means clustering. 
        It is defined as the sum of the squared distances between each data point and its assigned centroid. 
        The objective of K-means clustering is to minimize the inertia, which implies that we want to find the clustering solution that leads to the smallest sum of squared distances between data points and their assigned centroids.

        ## The inertia can be calculated as follows:

        1. For each cluster, calculate the sum of the squared distances between the data points and their assigned centroid.
        2. Sum the results from step 1 over all clusters.

        The k-means algorithm tries to minimize the inertia by iteratively re-assigning data points to their nearest centroid and recalculating the centroids until convergence.

        One limitation of using inertia as a measure of clustering quality is that it tends to decrease as the number of clusters increases, regardless of whether the additional clusters actually represent meaningful partitions of the data. Therefore, it is often recommended to use other measures, such as the silhouette score or the gap statistic, to determine the optimal number of clusters.
        """
    )
    st.image(
        "https://miro.medium.com/v2/resize:fit:880/1*xOGY4uu6ng7E8lPLP-onWw.png",
        width=600,
        caption="Inertia"
    )
    st.header("Elbow Method to choose The Best Value Of K")
    sns.lineplot(x=range(2, 13), y=inertia_scores)
    plt.title("Inertia")
    plt.xlabel("# Clusters")
    plt.ylabel("Inertia Score")
    plt.show()
    st.pyplot()
    st.header('3. Silhouette')
    st.write(
        """
        Silhouette is a metric used to evaluate the quality of clustering solutions. It measures how well each data point fits into its assigned cluster by computing the average distance between a data point and all other points in the same cluster (intra-cluster distance) and the average distance between the data point and all points in the nearest neighboring cluster (inter-cluster distance). The silhouette score ranges from -1 to 1, where a score of 1 indicates that the data point is very well-matched to its assigned cluster, a score of 0 indicates that the data point is equally close to two different clusters, and a score of -1 indicates that the data point is better matched to the neighboring cluster than to its assigned cluster.
        ## The silhouette score can be calculated as follows:
        1. For each data point, calculate its average distance to all other points in its assigned cluster (intra-cluster distance).
        2. For each data point, calculate its average distance to all points in the nearest neighboring cluster (inter-cluster distance).
        3. For each data point, calculate its silhouette score as (b - a) / max(a, b), where a is the intra-cluster distance and b is the inter-cluster distance.
        4. Calculate the average silhouette score across all data points in the dataset.

        A high silhouette score indicates that the clustering solution is appropriate, whereas a low score suggests that the clusters may be poorly defined or that the data points are better suited to other clusters.

        In practice, the silhouette score is often used as a secondary metric to evaluate the quality of clustering solutions in addition to the primary metric, such as inertia for K-means clustering. It can also be used to compare different clustering algorithms or to select the optimal number of clusters by comparing the silhouette scores for different values of k.
        """
    )
    visualizer = KElbowVisualizer(kmeans_model, k=(2, 12), metric='silhouette')
    visualizer.fit(wine_df_scaled_df)
    visualizer.show()
    st.pyplot()
    st.header("Plot The Data After Clustering Depending On the Number Of Clustering")
    k = st.slider(
        label="Choose How many Clusters You Need : ",
        min_value=2,
        max_value=12,
        value=4,
        step=1
    )
    final_k_mean_model = KMeans(n_clusters=k, random_state=42)
    final_k_mean_model.fit(wine_df_scaled_df)
    centroids = final_k_mean_model.cluster_centers_
    pca = PCA(n_components=2, random_state=42)
    new_features = pca.fit_transform(wine_df_scaled_df)
    sns.scatterplot(
        x=new_features[:, 0],
        y=new_features[:, 1],
        hue=final_k_mean_model.labels_,
        palette='light:#5A9'
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )
    plt.title("KMeans Clustering when k = {}".format(k))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    st.pyplot()
elif user_request == main_parts[2]:
    st.header('DBSCAN Clustering')
    st.write(
        """
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular clustering algorithm in machine learning and data mining. It is used to group together data points that are close to each other based on their density in the data space.
        The DBSCAN algorithm works by identifying "core" points, which are points that have a minimum number of other points within a specified distance (called the "epsilon" parameter), and "border" points, which are points that are within epsilon distance of a core point but do not have enough neighbors to be considered a core point. Points that are not core or border points are considered "noise" points.
        The algorithm begins by selecting an arbitrary unvisited point and finding all its neighbors within epsilon distance. If the point has at least the minimum number of neighbors, it is considered a core point, and a new cluster is formed. The algorithm then recursively adds all neighboring points that are also core points to the same cluster. Once all core points have been added to the cluster, the algorithm proceeds to the next unvisited point and repeats the process until all points have been visited.
        DBSCAN has several advantages over other clustering algorithms, such as its ability to handle clusters of arbitrary shape and its robustness to noise. However, it does require careful selection of the epsilon and minimum number of neighbors parameters, and it can be sensitive to the scaling of the data.
        """
    )
    st.header("DBSCAN Steps: ")
    st.write(
        """
        1. Select an arbitrary unvisited point from the dataset.
        2. Find all points within a specified distance (epsilon) from the selected point, forming a region called the "epsilon-neighborhood" of the point.
        3. If the number of points in the epsilon-neighborhood is greater than or equal to a specified minimum number of points (minPts), the selected point is considered a "core point".
        4. Create a new cluster and add the core point to it.
        5. Add all the points in the epsilon-neighborhood to the cluster, recursively repeating steps 2-4 for each point in the neighborhood that is also a core point.
        6. If the selected point is not a core point but is within the epsilon-neighborhood of some other core point, it is considered a "border point". Add it to the current cluster, but do not expand the cluster further.
        7. If the selected point is neither a core point nor a border point, it is considered a "noise point" and is ignored.
        8. Repeat steps 1-7 for all unvisited points in the dataset until all points have been visited.
        """
    )
    st.header("Choose eps value and minimum points")
    eps = st.slider(
        label="Choose The eps Value: ",
        min_value=0.5,
        max_value=5.0,
        step=0.1,
        value=2.0
    )
    min_points = st.slider(
        label="Choose The minimum Points: ",
        min_value=1,
        max_value=15,
        step=1,
        value=6
    )
    dbscan_model = DBSCAN(eps=eps, min_samples=min_points)
    dbscan_model.fit(wine_df_scaled_df)
    pca = PCA(n_components=2, random_state=42)
    new_features = pca.fit_transform(wine_df_scaled_df)
    sns.scatterplot(
        x=new_features[:, 0],
        y=new_features[:, 1],
        hue=dbscan_model.labels_,
        palette='light:#5A9'
    )
    plt.title("DBSCAN Clustering when eps = {} and minimum points = {}".format(
        eps, min_points))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    st.pyplot()

st.sidebar.header("Designed By: Hassan Elhefny")

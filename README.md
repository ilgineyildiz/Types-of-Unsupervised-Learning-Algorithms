# Iris Dataset Clustering Project
This project applies several clustering algorithms to the Iris dataset, including K-Means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models (GMM). The results are evaluated using metrics such as Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

## Project Overview
The Iris dataset is a classic dataset in machine learning, consisting of 150 samples of iris flowers with four features each (sepal length, sepal width, petal length, and petal width). The goal of this project is to cluster the samples into three groups corresponding to the three species: Setosa, Versicolour, and Virginica.

### 1. Data Preparation
The Iris dataset is loaded using sklearn.datasets.load_iris.
The dataset is scaled using StandardScaler for better performance of the clustering algorithms.
Principal Component Analysis (PCA) is used to reduce the dataset to two dimensions for visualization.

### 2. Clustering Algorithms
K-Means: A classic centroid-based clustering algorithm.
Hierarchical Clustering: An algorithm that builds a hierarchy of clusters.
DBSCAN: A density-based clustering algorithm.
Gaussian Mixture Models (GMM): A probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions.

### 3. Evaluation Metrics
Adjusted Rand Index (ARI): Measures the similarity between the true labels and the clustering labels, adjusted for chance.
Normalized Mutual Information (NMI): Measures the mutual dependence between the true labels and the clustering labels.

###4. Results
The performance of each clustering algorithm is as follows:
K-Means: ARI = 0.620, NMI = 0.659
Hierarchical Clustering: ARI = 0.586, NMI = 0.643
DBSCAN: ARI = 0.523, NMI = 0.615
GMM: ARI = 0.716, NMI = 0.735

### 5. Visualization
The results of the clustering algorithms are visualized using scatter plots of the two principal components. Each plot shows how the data points are clustered by the respective algorithm.

### Dependencies
Python 3.x
scikit-learn
Matplotlib
Seaborn
NumPy

### Running the Project
Clone the repository:
git clone https://github.com/yourusername/iris-clustering.git
cd iris-clustering

Install the required dependencies:
pip install -r requirements.txt

Run the Python script:
python iris_clustering.py

### Conclusion
This project demonstrates the effectiveness of various clustering algorithms on the Iris dataset. The Gaussian Mixture Model (GMM) achieved the highest performance among the tested algorithms, as indicated by the ARI and NMI scores.

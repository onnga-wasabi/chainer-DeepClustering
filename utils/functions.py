from sklearn.cluster import KMeans


def DeepClusteringKMeans(X):
    km = KMeans()
    km.fit(X)
    return km

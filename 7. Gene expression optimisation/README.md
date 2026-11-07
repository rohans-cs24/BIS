# Gene Expression Alogrithm
## Problem statement
https://pmc.ncbi.nlm.nih.gov/articles/PMC2632677  
**Tumor Clustering**  
To identify different types or subtypes of tumors, understand their similarities, and support cancer research and diagnosis.  

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load gene expression dataset
# Rows = tumor samples
# Columns = gene expression values

data = pd.read_csv("tumor_gene_expression.csv")

print("Dataset:")
print(data.head())

# Step 2: Remove sample ID column if present
if "Sample_ID" in data.columns:
    sample_ids = data["Sample_ID"]
    data = data.drop(columns=["Sample_ID"])
else:
    sample_ids = data.index

# Step 3: Handle missing values
data = data.apply(pd.to_numeric, errors="coerce")
data = data.fillna(data.median())

# Step 4: Select top 1000 variable genes
top_genes = data.var().nlargest(
    min(1000, data.shape[1])
).index

data = data[top_genes]

# Step 5: Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Step 6: Find the best number of clusters
best_score = -1
best_k = 2

for k in range(2, min(7, len(data))):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    print("Clusters:", k, "Score:", score)

    if score > best_score:
        best_score = score
        best_k = k

# Step 7: Apply K-Means clustering
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# Step 8: Display cluster assignments
results = pd.DataFrame({
    "Sample_ID": sample_ids,
    "Tumor_Cluster": clusters
})

print("\nTumor Clustering Results:")
print(results)

results.to_csv("tumor_cluster_results.csv", index=False)

# Step 9: Reduce dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 10: Visualize clusters
plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=clusters,
    palette="viridis",
    s=80
)

plt.title("Tumor Clustering Using Gene Expression")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Tumor Cluster")
plt.show()

print("\nBest number of clusters:", best_k)
print("Silhouette Score:", best_score)
```
<img width="776" height="547" alt="image" src="https://github.com/user-attachments/assets/d281f993-ffd1-4035-baa1-3e5656185a48" />

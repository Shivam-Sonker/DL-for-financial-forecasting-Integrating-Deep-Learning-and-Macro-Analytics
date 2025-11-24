import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors, BallTree


class MyDBSCAN:

    def __init__(self):
        self.stock_files = {
            "NVDA": "NVDA_data.csv",
            "VOO": "VOO_data.csv",
            "JPM": "JPM_data.csv",
            "MS": "MS_data.csv",
            "TSLA": "TSLA_data.csv",
            "AMD": "AMD_data.csv",
            "F": "F_data.csv",
            "AMZN": "AMZN_data.csv",
            "GOOG": "GOOG_data.csv",
            "INTC": "INTC_data.csv",
        }

        # Balance sheet
        self.balance_sheet_path = "./dataset/stock_balance_sheet.csv"
        self.balance_sheet_data = pd.read_csv(
            self.balance_sheet_path, parse_dates=["DATE"], index_col="DATE"
        )
        self.balance_sheet_quarterly = self.balance_sheet_data.resample("Q").last()

        # Quarterly Returns
        self.quarterly_returns_all_stocks = pd.DataFrame()

        for symbol, filename in self.stock_files.items():
            path = f"./dataset/stock_data/{filename}"
            stock_df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

            daily_returns = stock_df["Close"].pct_change().dropna()
            quarterly_returns = daily_returns.resample("Q").mean()

            self.quarterly_returns_all_stocks[symbol] = quarterly_returns

        self.quarterly_returns_all_stocks.dropna(inplace=True)

        self.financial_metrics = [
            "Revenue Growth (YoY)",
            "Shares Change",
            "Gross Margin",
            "Operating Margin",
            "Profit Margin",
            "Free Cash Flow Margin",
            "EBITDA Margin",
            "EBIT Margin",
            "Cash Growth",
            "Debt Growth",
        ]

        self.data = None
        self.features = None
        self.features_scaled = None
        self.scaler = StandardScaler()

        self.metrics_without_pca = {"eps": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
        self.metrics_with_pca = {"eps": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}

    # -------------------------------------------------------
    def set_stock_data(self):
        stock_data = {}
        for symbol in self.stock_files:
            df = pd.read_csv(f"./dataset/stock_data/{symbol}_data.csv", usecols=["Date", "Close"])
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            stock_data[symbol] = df

        df_close = pd.DataFrame({s: d["Close"] for s, d in stock_data.items()})
        df_close = df_close.ffill().dropna()

        self.data = df_close.pct_change().dropna()

    # -------------------------------------------------------
    def volatility_feature_engineering(self):
        """
        Builds volatility + PCA features.
        """
        returns = self.data.T
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(returns)

        self.features = pd.DataFrame(pca_data, index=returns.index, columns=["PC1", "PC2"])
        self.features["mean_returns"] = self.data.mean().values
        self.features["volatility"] = self.data.std().values

        self.features_scaled = self.scaler.fit_transform(self.features)

    # -------------------------------------------------------
    def safe_silhouette(self, labels, X):
        """
        Prevents silhouette crash.
        """
        unique = set(labels)
        if len(unique) <= 1:
            print("[WARNING] DBSCAN found only 1 cluster or all noise. Skipping silhouette.")
            return None
        return silhouette_score(X, labels)

    # -------------------------------------------------------
    def get_elbow_plot(self):
        tree = BallTree(self.features_scaled, leaf_size=2)
        k = 4
        dist, _ = tree.query(self.features_scaled, k=k)
        sorted_dist = np.sort(dist[:, k - 1])

        plt.plot(sorted_dist)
        plt.title("DBSCAN Elbow Plot")
        plt.xlabel("Points")
        plt.ylabel("Distance (k-th NN)")
        plt.grid(True)
        plt.show()

    # -------------------------------------------------------
    def volatility_clustering_without_pca(self):
        print("Clustering WITHOUT PCA")

        eps_values = np.arange(1.25, 1.75, 0.25)

        for eps in eps_values:
            model = DBSCAN(eps=eps, min_samples=3).fit(self.features_scaled)
            labels = model.labels_
            self.features["cluster"] = labels

            sil = self.safe_silhouette(labels, self.features_scaled)
            if sil is None:
                continue  # Skip invalid runs

            dav = davies_bouldin_score(self.features_scaled, labels)
            cal = calinski_harabasz_score(self.features_scaled, labels)

            self.metrics_without_pca["eps"].append(eps)
            self.metrics_without_pca["silhouette"].append(sil)
            self.metrics_without_pca["davies_bouldin"].append(dav)
            self.metrics_without_pca["calinski_harabasz"].append(cal)

            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=self.features, x="mean_returns", y="volatility",
                            hue="cluster", palette="viridis", s=80)
            plt.title(f"DBSCAN Clusters (eps={eps})")
            plt.show()

    # -------------------------------------------------------
    def volatility_clustering_with_pca(self):
        print("Clustering WITH PCA")

        eps_values = np.arange(0.75, 1.25, 0.25)

        for eps in eps_values:
            model = DBSCAN(eps=eps, min_samples=3).fit(self.features_scaled)
            labels = model.labels_
            self.features["cluster"] = labels

            sil = self.safe_silhouette(labels, self.features_scaled)
            if sil is None:
                continue

            dav = davies_bouldin_score(self.features_scaled, labels)
            cal = calinski_harabasz_score(self.features_scaled, labels)

            self.metrics_with_pca["eps"].append(eps)
            self.metrics_with_pca["silhouette"].append(sil)
            self.metrics_with_pca["davies_bouldin"].append(dav)
            self.metrics_with_pca["calinski_harabasz"].append(cal)

            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=self.features, x="PC1", y="PC2",
                            hue="cluster", palette="viridis", s=80)
            plt.title(f"DBSCAN Clusters with PCA (eps={eps})")
            plt.show()

    # -------------------------------------------------------
    def get_nearest_neighbors(self):
        nbrs = NearestNeighbors(n_neighbors=4)
        nbrs.fit(self.features_scaled)
        distances, _ = nbrs.kneighbors(self.features_scaled)

        distances = np.sort(distances[:, 1])

        plt.plot(distances)
        plt.title("KNN Distance Plot")
        plt.xlabel("Points")
        plt.ylabel("Distance")
        plt.show()

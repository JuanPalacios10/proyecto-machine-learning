from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from src.dataset.dataset import PokemonDataset
import plotly.express as px

def create_cluster():
    pokemon_dataset = PokemonDataset()
    df = pokemon_dataset.get_columns(
        [
            "Attack",
            "Defense",
            "HP",
        ]
    )

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(scaled_data)

    df["cluster"] = kmeans.labels_

    # fig = px.scatter_3d(
    #     df,
    #     x="Attack",
    #     y="Defense",
    #     z="HP",
    #     color="cluster",
    #     title="Pokemon Clusters based on Attack and Defense",
    # )
    # fig.show()

    for c in df['cluster'].unique():
        cluster_data = df[df['cluster'] == c]
        print(f"\nCluster {c}:")
        print(cluster_data.describe())

if __name__ == "__main__":
    create_cluster()

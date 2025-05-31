from typing import Callable, Self
from pandas import DataFrame, Series, pandas
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from src.dataset.dataset import PokemonDataset, PokemonStats
import plotly.express as px


class Clusters:
    def __init__(self, n_clusters: int, columns: list[str]) -> None:
        self.__n_clusters = n_clusters
        self.__clusters = list(range(n_clusters))
        self.__columns = columns
        self.__stats = DataFrame()
        self.__pokemons_group = PokemonDataset().get_columns(columns=columns).copy()

    def __graphing_3d_cluster(
        self, stats: DataFrame, x: str, y: str, z: str, color: str, title: str
    ) -> None:
        fig = px.scatter_3d(stats, x=x, y=y, z=z, color=color, title=title)
        fig.show()

    def __graphing_2d_cluster(
        self, stats: DataFrame, x: str, y: str, color: str, title: str
    ) -> None:
        fig = px.scatter(stats, x=x, y=y, color=color, title=title)
        fig.show()

    def show_graphics_clusters(self) -> Self:
        if len(self.__clusters) == 0:
            print("No se encontraron clusters. No es posible graficar.")
            return self

        columns = self.__stats.drop(columns="cluster").columns.tolist()
        title = f"Pokemos agrupados por Cluster de acuerdo a {', '.join(columns)}"

        if len(columns) == 2:
            self.__graphing_2d_cluster(
                stats=self.__stats,
                x=columns[0],
                y=columns[1],
                color="cluster",
                title=title,
            )
            return self

        if len(columns) == 3:
            self.__graphing_3d_cluster(
                stats=self.__stats,
                x=columns[0],
                y=columns[1],
                z=columns[2],
                color="cluster",
                title=title,
            )
            return self

        raise ValueError(
            "Graphing Cluster is only supported for 2 or 3 columns. "
            f"Received {len(columns)} columns."
        )

    def show_histogram(self) -> Self:
        numeric_cols = self.__stats.select_dtypes(include="number").columns.drop(
            "cluster"
        )

        for col in numeric_cols:
            fig = px.box(
                self.__stats,
                x="cluster",
                y=col,
                points="all",
                title=f"Distribución de '{col}' por Clúster",
            )
            fig.show()

        return self

    def __show_cluster_information(
        self,
        stats: DataFrame,
        clusters: list[int],
        show: Callable[[int, Series | DataFrame], None],
    ) -> None:
        if len(clusters) == 0:
            print("No se encontraron clusters. No es posible mostrar información.")
            return

        for cluster in clusters:
            cluster_data = stats[stats["cluster"] == cluster].drop(columns="cluster")
            show(cluster, cluster_data)

    def show_all_cluster_statistics(self) -> Self:
        def show_statistics(cluster: int, cluster_data: Series | DataFrame) -> None:
            print(f"\nEstadísticas del cluster {cluster}:")
            print(cluster_data.describe())

        self.__show_cluster_information(
            stats=self.__pokemons_group,
            clusters=self.__clusters,
            show=show_statistics,
        )

        return self

    def show_pokemons_in_cluster(
        self,
        n_pokemons: int = 10,
    ) -> Self:
        def show_pokemon_stats(cluster: int, cluster_data: Series | DataFrame) -> None:
            print(f"\nPrimeros {n_pokemons} pokemons del cluster {cluster}:")
            print("=" * 150)
            print("")
            print(cluster_data.head(n_pokemons))

        pokemons_groups = PokemonDataset().get_all_pokemons()
        pokemons_groups["cluster"] = self.__pokemons_group["cluster"].values
        pokemons_groups.columns = [
            "Pokedex" if col == PokemonStats.POKEDEX.value else col
            for col in pokemons_groups.columns
        ]

        self.__show_cluster_information(
            stats=pokemons_groups,
            clusters=self.__clusters,
            show=show_pokemon_stats,
        )

        return self

    def filter_better_clusters(self, query: str) -> Self:
        def filter_stats(stats: DataFrame, filtered_clusters: list[int]) -> DataFrame:
            filter_stats = DataFrame()

            for cluster in filtered_clusters:
                filtered_cluster = stats[stats["cluster"] == cluster]
                filter_stats = pandas.concat([filter_stats, filtered_cluster])

            if isinstance(filter_stats, Series):
                raise ValueError("filter_stats must be a DataFrame, not a Series.")

            return filter_stats

        cluster_means = self.__pokemons_group.groupby("cluster").mean()
        filtered_clusters_group = cluster_means.query(query)

        if filtered_clusters_group.empty:
            print("No se encontraron clusters que cumplan con la condición.")
            self.__clusters = []
            return self

        filtered_clusters = filtered_clusters_group.index.tolist()
        self.__pokemons_group = filter_stats(
            stats=self.__pokemons_group, filtered_clusters=filtered_clusters
        )
        self.__stats = filter_stats(
            stats=self.__stats, filtered_clusters=filtered_clusters
        )
        self.__clusters = filtered_clusters
        print(f"\nClusters filtrados {', '.join(map(str, filtered_clusters))}")

        return self

    def create_cluster(self, stats: DataFrame) -> Self:
        self.__stats = stats.copy()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(stats)

        kmeans = KMeans(n_clusters=self.__n_clusters, random_state=42)
        kmeans.fit(scaled_data)

        self.__pokemons_group["cluster"] = kmeans.labels_
        self.__stats["cluster"] = kmeans.labels_

        return self

    def refresh_clusters(self) -> None:
        self.__pokemons_group = (
            PokemonDataset().get_columns(columns=self.__columns).copy()
        )
        self.__stats = DataFrame()

    def create_filter_clusters(
        self, stats: DataFrame, range_clusters: range, query: str
    ) -> None:
        if range_clusters.stop - 1 != self.__n_clusters:
            raise ValueError("The stop value of range_clusters must match n_clusters.")

        n_clusters = range_clusters.stop - 1
        for cluster in range_clusters:
            print("")
            print("=" * 80)
            print(
                f"\nCreando clusters {cluster} de un total de {n_clusters} clusters..."
            )

            self.refresh_clusters()
            self.__n_clusters = cluster
            self.__clusters = list(range(cluster))
            self.create_cluster(stats=stats).filter_better_clusters(
                query
            ).show_all_cluster_statistics()

    def create_clusters(self, stats: DataFrame, range_clusters: range) -> None:
        if range_clusters.stop - 1 != self.__n_clusters:
            raise ValueError("The stop value of range_clusters must match n_clusters.")

        n_clusters = range_clusters.stop - 1
        for cluster in range_clusters:
            print("")
            print("=" * 80)
            print(
                f"\nCreando clusters {cluster} de un total de {n_clusters} clusters..."
            )

            self.refresh_clusters()
            self.__n_clusters = cluster
            self.__clusters = list(range(cluster))
            self.create_cluster(stats=stats).show_all_cluster_statistics()

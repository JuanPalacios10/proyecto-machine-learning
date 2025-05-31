from pandas import DataFrame
from src.dataset.dataset import PokemonDataset, PokemonStats
from src.implementation.clusters import Clusters


def main() -> None:
    columns = [
        PokemonStats.HP.value,
        PokemonStats.ATTACK.value,
        PokemonStats.DEFENSE.value,
        PokemonStats.SPECIAL_ATTACK.value,
        PokemonStats.SPECIAL_DEFENSE.value,
        PokemonStats.SPEED.value,
    ]
    total_stats = PokemonDataset().get_all_stats().copy()
    stats = DataFrame()

    stats["BalanceScore"] = 1 / (total_stats.std(axis=1) + 1e-5)

    stats["Tanks"] = (
        total_stats[PokemonStats.HP.value]
        + total_stats[PokemonStats.SPECIAL_DEFENSE.value]
    ) / (total_stats[PokemonStats.SPEED.value] + total_stats[PokemonStats.ATTACK.value])

    stats["GlassCannon"] = total_stats[PokemonStats.SPECIAL_ATTACK.value] / (
        total_stats[PokemonStats.DEFENSE.value] + total_stats[PokemonStats.HP.value]
    )

    clusters = Clusters(n_clusters=9, columns=columns)
    clusters.create_cluster(
        stats=stats
    ).show_all_cluster_statistics().show_histogram().show_graphics_clusters()


if __name__ == "__main__":
    main()

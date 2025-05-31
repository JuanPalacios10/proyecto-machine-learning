from enum import Enum
import pandas as pd
import os


class PokemonStats(str, Enum):
    POKEDEX = "#"
    NAME = "Name"
    TYPE_1 = "Type 1"
    TYPE_2 = "Type 2"
    TOTAL = "Total"
    HP = "HP"
    ATTACK = "Attack"
    DEFENSE = "Defense"
    SPECIAL_ATTACK = "Sp. Atk"
    SPECIAL_DEFENSE = "Sp. Def"
    SPEED = "Speed"
    GENERATION = "Generation"
    LEGENDARY = "Legendary"


class SingletonMeta(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            instance = super().__call__(*args, **kwargs)
            cls.__instances[cls] = instance

        return cls.__instances[cls]


class PokemonDataset(metaclass=SingletonMeta):
    __pokemon_dataset = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "Pokemon.csv")
    )

    def get_columns(self, columns: list[str]) -> pd.DataFrame:
        return self.__pokemon_dataset.loc[:, columns]

    def get_all_stats(self) -> pd.DataFrame:
        return self.get_columns(
            [
                PokemonStats.HP.value,
                PokemonStats.ATTACK.value,
                PokemonStats.DEFENSE.value,
                PokemonStats.SPECIAL_ATTACK.value,
                PokemonStats.SPECIAL_DEFENSE.value,
                PokemonStats.SPEED.value,
            ]
        )

    def get_all_pokemons(self) -> pd.DataFrame:
        return self.__pokemon_dataset.copy()

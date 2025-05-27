import pandas as pd
import os


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

    def get_all_pokemons(self) -> pd.DataFrame:
        return self.__pokemon_dataset

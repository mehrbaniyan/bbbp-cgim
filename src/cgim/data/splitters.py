from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


class Splitter:
    """Base class for DataFrame splitters."""
    def split(self,
             df: pd.DataFrame,
             frac_train: float = 0.8,
             frac_valid: float = 0.1,
             frac_test: float = 0.1,
             smile_column: str = 'smiles',
             seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class RandomSplitter(Splitter):
    """Original RandomSplitter logic adapted for DataFrames"""
    def split(self,
              df: pd.DataFrame,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              smile_column: str = 'smiles',
              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        if seed is not None:
            np.random.seed(seed)

        num_datapoints = len(df)
        train_cutoff = int(frac_train * num_datapoints)
        valid_cutoff = int((frac_train + frac_valid) * num_datapoints)

        shuffled = np.random.permutation(range(num_datapoints))
        return (shuffled[:train_cutoff],
                shuffled[train_cutoff:valid_cutoff],
                shuffled[valid_cutoff:])


class ScaffoldSplitter(Splitter):
    """Original ScaffoldSplitter logic adapted for DataFrames"""
    def split(self,
              df: pd.DataFrame,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              smile_column: str = 'smiles',
              include_chirality: bool = False,
              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
        scaffold_sets = self.generate_scaffolds(df, smile_column, include_chirality=include_chirality)

        train_cutoff = frac_train * len(df)
        valid_cutoff = (frac_train + frac_valid) * len(df)

        train_inds: List[int] = []
        valid_inds: List[int] = []
        test_inds: List[int] = []

        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                    test_inds.extend(scaffold_set)
                else:
                    valid_inds.extend(scaffold_set)
            else:
                train_inds.extend(scaffold_set)

        return (np.array(train_inds),
                np.array(valid_inds),
                np.array(test_inds))

    def generate_scaffolds(self, df: pd.DataFrame, smile_column: str, include_chirality: bool = False) -> List[List[int]]:
        """Original scaffold generation logic adapted for DataFrames"""
        scaffolds = {}
        for idx, smiles in enumerate(df[smile_column]):
            try:
                mol = Chem.MolFromSmiles(smiles)

                scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
                if scaffold is not None:
                    if scaffold not in scaffolds:
                        scaffolds[scaffold] = [idx]
                    else:
                        scaffolds[scaffold].append(idx)
            except:
                continue

        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set
            for (scaffold,
                 scaffold_set) in sorted(scaffolds.items(),
                                         key=lambda x: (len(x[1]), x[1][0]),
                                         reverse=True)
        ]
        return scaffold_sets


class RandomScaffoldSplitter:
    """Splitter that randomly permutes scaffold sets and assigns them to train, valid, and test splits"""
    def split(self,
              df: pd.DataFrame,
              frac_train: float = 0.8,
              frac_valid: float = 0.1,
              frac_test: float = 0.1,
              smile_column: str = 'smiles',
              seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        if seed is not None:
            np.random.seed(seed)

        scaffold_sets = self.generate_scaffolds(df, smile_column)
        permuted_indices = np.random.permutation(len(scaffold_sets))
        permuted_scaffold_sets = [scaffold_sets[i] for i in permuted_indices]
        n_total_valid = int(np.floor(frac_valid * len(df)))
        n_total_test = int(np.floor(frac_test * len(df)))
        train_idx: List[int] = []
        valid_idx: List[int] = []
        test_idx: List[int] = []

        for scaffold_set in permuted_scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return (np.array(train_idx), np.array(valid_idx), np.array(test_idx))

    def generate_scaffolds(self, df: pd.DataFrame, smile_column: str) -> List[List[int]]:
        scaffolds: dict = {}
        for idx, smiles in enumerate(df[smile_column]):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = [idx]
                else:
                    scaffolds[scaffold].append(idx)
            except:
                continue
        return list(scaffolds.values())


def make_splitter(name: str):
    splitters = {
        "random": RandomSplitter,
        "scaffold": ScaffoldSplitter,
        "random_scaffold": RandomScaffoldSplitter,
    }
    try:
        return splitters[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown split '{name}'. Choose from {sorted(splitters)}") from exc


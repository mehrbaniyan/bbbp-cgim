from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(image_size: Tuple[int, int] = (224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.expand(3, -1, -1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class Molecule3DDatasetUSPre(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 image_size: Tuple[int, int] = (224, 224),
                 transform: Optional[transforms.Compose] = None,
                 is_validation: bool = False,
                 label_column1: str = 'Class',
                 label_column2: str = 'Class',
                 smiles_column: str = 'SMILES',
                 svd_components: Optional[TruncatedSVD] = None,
                 scaffold_fp_radius: int = 2,
                 scaffold_fp_bits: int = 2048,
                 scaffold_svd_dim: int = 16):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform or transforms.ToTensor()
        self.is_validation = is_validation

        unique_labels = sorted(self.df[label_column1].unique())
        self.label_to_idx1 = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels1 = torch.tensor(
            [self.label_to_idx1[label] for label in self.df[label_column1]],
            dtype=torch.long
        )
        unique_labels2 = sorted(self.df[label_column2].unique())
        self.label_to_idx2 = {label: idx for idx, label in enumerate(unique_labels2)}
        self.labels2 = torch.tensor(
            [self.label_to_idx2[label] for label in self.df[label_column2]],
            dtype=torch.long
        )

        self.smiles_column = smiles_column
        if smiles_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain a '{smiles_column}' column for SMILES strings.")

        self.scaffold_fp_radius = scaffold_fp_radius
        self.scaffold_fp_bits = scaffold_fp_bits
        self.scaffold_svd_dim = scaffold_svd_dim

        if svd_components is not None:
            self.svd = svd_components
            if "scaffold_svd" not in self.df.columns:
                raise ValueError("When svd_components is provided, DataFrame must have 'scaffold_svd' column.")
            self.scaffold_svd_matrix = np.stack(self.df["scaffold_svd"].to_list(), axis=0)
        else:
            if self.is_validation:
                raise ValueError("Validation mode requires a pre-fitted svd_components and 'scaffold_svd' column.")
            from scipy import sparse
            raw_fps = []
            for smi in self.df[self.smiles_column].tolist():
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raw_fps.append(sparse.csr_matrix((1, self.scaffold_fp_bits), dtype=np.float32))
                    continue
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.scaffold_fp_radius, nBits=self.scaffold_fp_bits)
                arr = np.zeros((self.scaffold_fp_bits,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                cols = np.nonzero(arr)[0]
                data = arr[cols]
                rows = np.zeros_like(cols)
                raw_fps.append(sparse.csr_matrix((data, (rows, cols)),
                                                 shape=(1, self.scaffold_fp_bits), dtype=np.float32))
            raw_matrix = sparse.vstack(raw_fps, format='csr')

            self.svd = TruncatedSVD(n_components=self.scaffold_svd_dim, n_iter=10, random_state=0)
            svd_vectors = self.svd.fit_transform(raw_matrix)
            scaler = StandardScaler()
            self.scaffold_svd_matrix = scaler.fit_transform(svd_vectors)
            self.df["scaffold_svd"] = list(self.scaffold_svd_matrix)

        self.scaffold_svd_matrix = self.scaffold_svd_matrix.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_dir = row['image_path']

        if pd.notna(mol_dir) and isinstance(mol_dir, str) and os.path.isdir(mol_dir):
            scaffold_path = os.path.join(mol_dir, 'scaffold.png')
            if os.path.exists(scaffold_path):
                scaffold_img = Image.open(scaffold_path).convert('RGB')
                scaffold_img_tensor = self.transform(scaffold_img)
            else:
                scaffold_img_tensor = self.transform(Image.new('RGB', self.image_size, (0, 0, 0)))
            image_paths = [
                os.path.join(mol_dir, f)
                for f in os.listdir(mol_dir)
                if f.endswith('.png') and f != 'scaffold.png'
            ]
        else:
            dummy = Image.new('RGB', self.image_size, (0, 0, 0))
            tensor1 = tensor2 = scaffold_img_tensor = self.transform(dummy)
            label1 = self.labels1[idx]
            label2 = self.labels2[idx]
            scaffold_pca_vec = torch.zeros(self.scaffold_svd_dim, dtype=torch.float32)
            return tensor1, tensor2, label1, label2, scaffold_img_tensor, scaffold_pca_vec

        if len(image_paths) == 0:
            dummy = Image.new('RGB', self.image_size, (0, 0, 0))
            tensor1 = tensor2 = scaffold_img_tensor = self.transform(dummy)
            label1 = self.labels1[idx]
            label2 = self.labels2[idx]
            scaffold_pca_vec = torch.zeros(self.scaffold_svd_dim, dtype=torch.float32)
            return tensor1, tensor2, label1, label2, scaffold_img_tensor, scaffold_pca_vec

        if self.is_validation:
            img1 = img2 = Image.open(image_paths[0]).convert('RGB')
        else:
            img1_path, img2_path = random.sample(image_paths, 2)
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

        tensor1 = self.transform(img1)
        tensor2 = self.transform(img2)
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]
        scaffold_pca_vec = torch.from_numpy(self.scaffold_svd_matrix[idx])
        return tensor1, tensor2, label1, label2, scaffold_img_tensor, scaffold_pca_vec


class Molecule3DDatasetUS(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 image_size: Tuple[int, int] = (224, 224),
                 transform: Optional[transforms.Compose] = None,
                 is_validation: bool = False,
                 label_column: Union[str, List[str]] = 'Class',
                 smiles_column: str = 'SMILES',
                 pca_components: Optional[PCA] = None,
                 scaffold_fp_radius: int = 2,
                 scaffold_fp_bits: int = 1024,
                 scaffold_pca_dim: int = 16):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform or transforms.ToTensor()
        self.is_validation = is_validation

        if isinstance(label_column, str):
            label_cols = [label_column]
        else:
            label_cols = list(label_column)
        self.label_column = label_column
        self._label_mappings = {}
        label_tensors = []
        for lbl in label_cols:
            if lbl not in self.df.columns:
                raise ValueError(f"Label column '{lbl}' not found in DataFrame.")
            unique_vals = sorted([v for v in self.df[lbl].unique() if v != -1])
            mapping = {v: i for i, v in enumerate(unique_vals)}
            self._label_mappings[lbl] = mapping
            mapped = [mapping[x] if x in mapping else -1 for x in self.df[lbl]]
            tensor_lbls = torch.tensor(mapped, dtype=torch.long)
            label_tensors.append(tensor_lbls)
        if len(label_tensors) == 1:
            self.labels = label_tensors[0]
        else:
            self.labels = torch.stack(label_tensors).t().contiguous()

        self.smiles_column = smiles_column
        if smiles_column not in self.df.columns:
            raise ValueError(f"DataFrame must contain a '{smiles_column}' column for SMILES strings.")

        self.scaffold_fp_radius = scaffold_fp_radius
        self.scaffold_fp_bits = scaffold_fp_bits
        self.scaffold_pca_dim = scaffold_pca_dim

        if pca_components is not None:
            self.pca = pca_components
            if "scaffold_pca" not in self.df.columns:
                raise ValueError("When pca_components is provided, DataFrame must have 'scaffold_pca' column.")
            self.scaffold_pca_matrix = np.stack(self.df["scaffold_pca"].to_list(), axis=0)
        else:
            if self.is_validation:
                raise ValueError("Validation mode requires a pre-fitted pca_components and 'scaffold_pca' column.")
            raw_fps = []
            for smi in self.df[self.smiles_column].tolist():
                raw_fps.append(self._scaffold_fp_vector(smi))
            raw_matrix = np.stack(raw_fps, axis=0)
            self.pca = PCA(n_components=self.scaffold_pca_dim)
            self.scaffold_pca_matrix = self.pca.fit_transform(raw_matrix)
            self.df["scaffold_pca"] = list(self.scaffold_pca_matrix)

        self.scaffold_pca_matrix = self.scaffold_pca_matrix.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_dir = row['image_path']
        dummy = Image.new('RGB', self.image_size, (0, 0, 0))

        if pd.notna(mol_dir) and isinstance(mol_dir, str) and os.path.isdir(mol_dir):
            scaffold_path = os.path.join(mol_dir, 'scaffold.png')
            if os.path.exists(scaffold_path):
                scaffold_img = Image.open(scaffold_path).convert('RGB')
                scaffold_img_tensor = self.transform(scaffold_img)
            else:
                scaffold_img_tensor = self.transform(dummy)
            image_paths = [
                os.path.join(mol_dir, f)
                for f in os.listdir(mol_dir)
                if f.endswith('.png') and f != 'scaffold.png'
            ]
        else:
            tensor1 = tensor2 = scaffold_img_tensor = self.transform(dummy)
            label = self.labels[idx]
            scaffold_pca_vec = torch.zeros(self.scaffold_pca_dim, dtype=torch.float32)
            return tensor1, tensor2, label, scaffold_img_tensor, scaffold_pca_vec

        if len(image_paths) == 0:
            tensor1 = tensor2 = scaffold_img_tensor
            label = self.labels[idx]
            scaffold_pca_vec = torch.zeros(self.scaffold_pca_dim, dtype=torch.float32)
            return tensor1, tensor2, label, scaffold_img_tensor, scaffold_pca_vec

        if self.is_validation:
            img1 = img2 = Image.open(image_paths[0]).convert('RGB')
        else:
            if len(image_paths) == 1:
                img1 = img2 = Image.open(image_paths[0]).convert('RGB')
            else:
                img1_path, img2_path = random.sample(image_paths, 2)
                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')

        tensor1 = self.transform(img1)
        tensor2 = self.transform(img2)
        label = self.labels[idx]
        scaffold_pca_vec = torch.from_numpy(self.scaffold_pca_matrix[idx])
        return tensor1, tensor2, label, scaffold_img_tensor, scaffold_pca_vec

    def _scaffold_fp_vector(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((self.scaffold_fp_bits,), dtype=np.int8)
        core_smiles = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False
        )
        core_mol = Chem.MolFromSmiles(core_smiles)
        if core_mol is None:
            return np.zeros((self.scaffold_fp_bits,), dtype=np.int8)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            core_mol,
            radius=self.scaffold_fp_radius,
            nBits=self.scaffold_fp_bits
        )
        arr = np.zeros((self.scaffold_fp_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr


def collate_pre(batch):
    view1, view2, labels1, labels2, scaffold_img, reg_vec = zip(*batch)
    return (
        torch.stack(view1),
        torch.stack(view2),
        torch.stack(labels1),
        torch.stack(labels2),
        torch.stack(scaffold_img),
        torch.stack(reg_vec)
    )


def collate_ft(batch):
    view1, view2, labels, scaffold_img, scaffold_pca = zip(*batch)
    return (
        torch.stack(view1),
        torch.stack(view2),
        torch.stack(labels),
        torch.stack(scaffold_img),
        torch.stack(scaffold_pca)
    )


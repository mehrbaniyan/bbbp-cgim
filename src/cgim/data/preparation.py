from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_image_paths(df, csv_path):
    """Resolve portable image paths stored relative to the repository root."""
    if "image_path" not in df.columns:
        raise ValueError("Processed CSV must contain an 'image_path' column")

    csv_parent = Path(csv_path).resolve().parent

    def resolve(value):
        if pd.isna(value) or not isinstance(value, str):
            return value
        path = Path(value)
        if path.is_absolute():
            return str(path)
        project_path = PROJECT_ROOT / path
        csv_path = csv_parent / path
        if csv_path.exists() and not project_path.exists():
            return str(csv_path)
        return str(project_path)

    df["image_path"] = df["image_path"].map(resolve)
    return df


def compute_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.int8)
    else:
        return None


def maccs_fp_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((167,), dtype=np.int8)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def full_fp_vector(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def ecfp_fp_vector(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return sparse.csr_matrix((1, nBits), dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    cols = np.nonzero(arr)[0]
    data = arr[cols]
    rows = np.zeros_like(cols)
    return sparse.csr_matrix((data, (rows, cols)), shape=(1, nBits), dtype=np.float32)


def prepare_downstream_frame(
    csv_path,
    smiles_column,
    fingerprint,
    feature_dim=16,
    drop_duplicate_smiles=False,
):
    """Run the repeated downstream CSV/MACCS/PCA blocks once."""
    df = pd.read_csv(csv_path)
    df = _resolve_image_paths(df, csv_path)
    if drop_duplicate_smiles:
        df = df.drop_duplicates(subset=[smiles_column])

    df['maccs'] = df[smiles_column].apply(compute_maccs)
    df = df[df['maccs'].notnull()].reset_index(drop=True)

    if fingerprint == "maccs_pca":
        raw = np.stack([maccs_fp_vector(smi) for smi in df[smiles_column]], axis=0)
    elif fingerprint == "full_ecfp_pca":
        raw = np.stack([full_fp_vector(smi) for smi in df[smiles_column]], axis=0)
    else:
        raise ValueError("fingerprint must be 'maccs_pca' or 'full_ecfp_pca'")

    pca = PCA(n_components=feature_dim)
    vectors = pca.fit_transform(raw).astype(np.float32)
    df['scaffold_pca'] = list(vectors)
    return df, pca


def prepare_pretrain_frame(
    csv_path,
    smiles_column="smiles",
    sample_size=10000,
    sample_seed=42,
    clusters=(10, 100),
    cluster_seed=42,
    fingerprint_bits=2048,
    radius=2,
    components=64,
    n_iter=10,
    svd_seed=0,
):
    """Run pretraining sampling, clustering, and SVD preparation."""
    df = pd.read_csv(csv_path)
    df = _resolve_image_paths(df, csv_path)
    if sample_size is not None:
        if sample_size > len(df):
            raise ValueError(f"sample_size={sample_size} exceeds the {len(df)} available rows")
        df = df.sample(n=sample_size, random_state=sample_seed)

    df['maccs'] = df[smiles_column].apply(compute_maccs)
    df = df[df['maccs'].notnull()].reset_index(drop=True)

    maccs_array = np.stack(df['maccs'].values)
    for k in clusters:
        actual_k = min(k, len(df))
        kmeans = KMeans(n_clusters=actual_k, random_state=cluster_seed)
        df[f'cluster_{k}'] = kmeans.fit_predict(maccs_array)

    ecfp_fps = [
        ecfp_fp_vector(smi, radius=radius, nBits=fingerprint_bits)
        for smi in df[smiles_column]
    ]
    ecfp_matrix = sparse.vstack(ecfp_fps, format='csr')
    svd = TruncatedSVD(n_components=components, n_iter=n_iter, random_state=svd_seed)
    ecfp_svd_vectors = svd.fit_transform(ecfp_matrix)
    scaler = StandardScaler()
    ecfp_pca_vectors = scaler.fit_transform(ecfp_svd_vectors).astype(np.float32)
    df['scaffold_svd'] = list(ecfp_pca_vectors)
    return df, svd, scaler


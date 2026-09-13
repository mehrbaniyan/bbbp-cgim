from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

RDLogger.DisableLog('rdApp.*')


class DatasetGenerator:
    """Offline RDKit/PyMOL renderer shared by pretraining and downstream data."""

    def __init__(self, input_csv: str, output_csv: str, base_folder: str, dataset_name: str,
                 smile_column: str = 'smiles', image_size: Tuple[int, int] = (224, 224),
                 valid: bool = False, n_conformers: int = 5, num_views: int = 120,
                 path_root: str | None = None):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.base_dir = os.path.join(base_folder, dataset_name)
        self.image_size = image_size
        self.valid = valid
        self.n_conformers = n_conformers
        self.num_views = 1 if valid else num_views
        self.path_root = Path(path_root).resolve() if path_root else None
        self.df = pd.read_csv(input_csv)
        self.smile_column = smile_column
        self._cmd = None
        os.makedirs(self.base_dir, exist_ok=True)

    def _start_pymol(self):
        if self._cmd is None:
            import pymol
            from pymol import cmd

            pymol.finish_launching(['pymol', '-cq'])
            cmd.feedback("disable", "all", "everything")
            self._cmd = cmd
        return self._cmd

    def generate_dataset(self):
        """Generate molecule images and update CSV with paths."""
        image_paths = []
        for idx, row in self.df.iterrows():
            smiles = row[self.smile_column]
            mol_dir = os.path.join(self.base_dir, f'molecule_{idx}')
            os.makedirs(mol_dir, exist_ok=True)

            if self._is_molecule_processed(mol_dir):
                image_paths.append(self._stored_path(mol_dir))
                continue

            try:
                self._process_molecule(smiles, mol_dir, idx)
                image_paths.append(self._stored_path(mol_dir))
            except Exception as e:
                print(f"Error processing molecule {idx}: {str(e)}")
                image_paths.append(None)

        self.df['image_path'] = image_paths
        Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.output_csv, index=False)
        return self.df

    def _stored_path(self, mol_dir: str) -> str:
        path = Path(mol_dir).resolve()
        if self.path_root is None:
            return str(path)
        try:
            return path.relative_to(self.path_root).as_posix()
        except ValueError:
            return str(path)

    def _is_molecule_processed(self, mol_dir: str) -> bool:
        """Check completion against the configured number of molecular views."""
        if not os.path.exists(mol_dir):
            return False
        png_files = [f for f in os.listdir(mol_dir) if f.endswith('.png')]
        return 'scaffold.png' in png_files and len(png_files) >= self.num_views + 1

    def _process_molecule(self, smiles: str, mol_dir: str, idx: int):
        print(f"-> Processing molecule {idx}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        try:
            original_mol = Chem.MolFromSmiles(smiles)
            if original_mol and original_mol.GetNumAtoms() > 0:
                scaffold_smiles = MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
                scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
                if scaffold_mol and scaffold_mol.GetNumAtoms() > 0:
                    if AllChem.EmbedMolecule(scaffold_mol, randomSeed=42) == 0:
                        AllChem.UFFOptimizeMolecule(scaffold_mol)
                        self._render_single_view(
                            scaffold_mol,
                            (0, 0, 0),
                            os.path.join(mol_dir, 'scaffold.png'),
                            None
                        )
        except Exception as e:
            print(f"Scaffold error for {idx}: {str(e)}")

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            raise ValueError("Embedding failed")
        AllChem.UFFOptimizeMolecule(mol)

        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=self.n_conformers, randomSeed=42)
        for conf_id in conf_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

        for img_idx in range(self.num_views):
            output_path = os.path.join(mol_dir, f'img_{img_idx}.png')
            if os.path.exists(output_path):
                continue

            aug_mol = Chem.Mol(mol)
            aug_mol = Chem.AddHs(aug_mol)

            if img_idx == 0:
                masked_indices = None
                rotation = (0, 0, 0)
            else:
                if len(conf_ids) > 0 and random.random() < 0.5:
                    conf_id = random.choice(list(conf_ids))
                    aug_mol.GetConformer(conf_id)

                masked_indices = self._get_masked_indices(aug_mol, 0.0, 0.15)
                rotation = (random.uniform(-15, 15),
                            random.uniform(-15, 15),
                            random.uniform(-15, 15))

            self._render_single_view(aug_mol, rotation, output_path, masked_indices)

    def _get_masked_indices(self, mol, min_fraction=0, max_fraction=0.25):
        valid_indices = [atom.GetIdx() for atom in mol.GetAtoms()
                         if atom.GetAtomicNum() != 1 or atom.GetDegree() > 0]
        fraction = random.uniform(min_fraction, max_fraction)
        num_to_mask = max(1, int(len(valid_indices) * fraction))
        return random.sample(valid_indices, num_to_mask)

    def apply_coordinate_noise(self, mol, noise_range=(-1.0, 1.0), fraction=0.08):
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        num_to_noise = max(1, int(num_atoms * fraction))
        indices = random.sample(range(num_atoms), num_to_noise)
        for idx in indices:
            pos = conf.GetAtomPosition(idx)
            noise = [random.uniform(*noise_range) for _ in range(3)]
            new_pos = (pos.x + noise[0], pos.y + noise[1], pos.z + noise[2])
            conf.SetAtomPosition(idx, new_pos)

    def _render_single_view(self, mol, rotation, output_path, masked_indices=None):
        cmd = self._start_pymol()
        sdf_path = Path("temp_mol.sdf")
        try:
            Chem.MolToMolFile(mol, str(sdf_path))
            cmd.reinitialize()
            cmd.load(str(sdf_path), "molecule")
            cmd.set("stick_ball", "off")
            cmd.show("sticks", "molecule")
            cmd.show("spheres", "molecule")
            cmd.set("sphere_scale", 0.25)
            cmd.set("ray_trace_mode", 0)
            cmd.set("ray_opaque_background", 1)
            cmd.set("transparency", 0)
            cmd.set("stick_transparency", 0)
            cmd.set("sphere_transparency", 0)
            cmd.set("depth_cue", 0)
            cmd.bg_color("black")
            cmd.orient("molecule")

            if masked_indices:
                for idx in masked_indices:
                    cmd.remove(f"molecule and index {idx+1}")

            elev, azim, zrot = rotation
            cmd.rotate("x", elev, "molecule")
            cmd.rotate("y", azim, "molecule")
            cmd.rotate("z", zrot, "molecule")
            cmd.zoom("molecule", 1.75)
            cmd.png(str(output_path), width=self.image_size[0],
                    height=self.image_size[1], dpi=300, ray=1, quiet=1)

            timeout = 10
            start = time.time()
            while not os.path.exists(output_path) and (time.time() - start) < timeout:
                time.sleep(0.1)

            if os.path.exists(output_path):
                img = Image.open(output_path).convert('RGB')
                img = img.resize(self.image_size)
                img.save(output_path)
            else:
                Image.new('RGB', self.image_size, (0, 0, 0)).save(output_path)
        finally:
            sdf_path.unlink(missing_ok=True)


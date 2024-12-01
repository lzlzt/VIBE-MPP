import random
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.data import Data


def get_atom_poses(mol, conf):   
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses


def get_steady_mol(smiles,num):
    try:
        mol = AllChem.MolFromSmiles(smiles)
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=num)
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        conf = new_mol.GetConformer(id=int(index))
    except:
        try:
            mol = AllChem.MolFromSmiles(smiles)
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(mol, numConfs=num,useRandomCoords=True)
            res = AllChem.MMFFOptimizeMoleculeConfs(mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            conf = mol.GetConformer(id=int(index))
        except:
            mol = AllChem.MolFromSmiles(smiles)
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            conf = new_mol.GetConformer(id=int(res))
    return new_mol,conf


def get_distance(mol,conf):
    num_atoms = mol.GetNumAtoms()
    distances = np.zeros((num_atoms, num_atoms))
    atom_positions = get_atom_poses(mol,conf)
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            dist_x = atom_positions[i][0] - atom_positions[j][0]
            dist_y = atom_positions[i][1] - atom_positions[j][1]
            dist_z = atom_positions[i][2] - atom_positions[j][2]
            dist = (dist_x**2 + dist_y**2 + dist_z**2) ** 0.5
            distances[i, j] = distances[j, i] = round(dist,3)
    return distances


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def get_virtual_mol(mol):
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atoms.append(atom)

    virtual_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in atoms:
        new_atom = copy_atom(atom)
        virtual_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if a1 in atoms and a2 in atoms:
            bt = bond.GetBondType()
            virtual_mol.AddBond(a1,a2,bt)
        else:
            break
    return virtual_mol

def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch


def is_ring(mol, atom_idx1, atom_idx2):
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    for ring in atom_rings:
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            if atom_idx1 in ring and atom_idx2 in ring:
                return True
    return False


def get_van_der_waals_force(r, sigma=3.4, epsilon=0.238):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


def get_bond_strength(distances):
    dim = distances.shape[0]
    van_der_waals_forces = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i+1,dim):
            van_der_waals_force = get_van_der_waals_force(distances[i][j])
            van_der_waals_forces[i][j] = van_der_waals_forces[j][i] = van_der_waals_force
    flattened_forces = van_der_waals_forces.flatten()
    strengths = pd.qcut(flattened_forces, q=10, labels=False,duplicates="drop") + 1  # 分类标签从1开始
    strengths = strengths.reshape(dim, dim)
    return strengths

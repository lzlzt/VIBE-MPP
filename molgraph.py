import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from utils import * 




threshold = 10       #原子间相互影响距离的阈值
conf_nums = 20      #设置生成原子构象的数量并从中筛选出最稳定的



class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol_graph = MolGraph(smiles) 
        return mol_graph

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),  #元素周期表序号
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],    #原子的手性
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [                          #表征化学键的类型
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information（双键的立体构型）
        Chem.rdchem.BondDir.NONE,               #（双键的立体构型没有特定方向）
        Chem.rdchem.BondDir.ENDUPRIGHT,         #双键中一个碳原子上的基团相对于另一个碳原子的基团处于向上的立体构型。
        Chem.rdchem.BondDir.ENDDOWNRIGHT        #向下
    ],
    'possible_bond_inring': [None, False, True]
}



class MolGraph(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol ,self.conf = get_steady_mol(smiles,conf_nums)

        atom_features_list = []
        for atom in self.mol.GetAtoms():
            atom_feature = [allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [allowable_features[
                                             'possible_degree_list'].index(atom.GetDegree())]

            atom_features_list.append(atom_feature)

        # 这里是非虚拟节点的特征
        self.x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(self.mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in self.mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = [allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
                    bond.IsInRing())]

                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            self.edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            self.edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                                  dtype=torch.long)
        else:
            self.edge_index_nosuper = torch.empty((2, 0), dtype=torch.long)  # edgeCOO索引，[[row索引],[col索引]]
            self.edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

        # add super node添加一个超级节点统领整个图
        num_atoms = self.x_nosuper.size(0)  # 张量第一个维度的大小，即原子的数量

        #最后连接整个图的一个超级节点
        super_x = torch.tensor([[119, 0]]).to(self.x_nosuper.device)
        virtual_mol = get_virtual_mol(self.mol)
        virtual_num_atoms = virtual_mol.GetNumAtoms()

        if virtual_num_atoms > 2:
            virtual_x = torch.tensor([[120, 0]]).repeat_interleave(virtual_num_atoms, dim=0).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, virtual_x, super_x), dim=0)

            virtual_edge_index = []

            distances = get_distance(self.mol,self.conf)
            n = min(virtual_num_atoms,num_atoms)
            for i in range(n):
                for j in range(i + 1, n):
                    atom_i = self.mol.GetAtomWithIdx(i)
                    atom_j = self.mol.GetAtomWithIdx(j)
                    bond = self.mol.GetBondBetweenAtoms(i, j)
                    dist = distances[i, j]
                    if bond is not None:
                        pass
                    else:
                        if dist > threshold:
                            pass
                        else:
                            virtual_edge_index = virtual_edge_index + [[i,num_atoms + j]] + [[j,num_atoms + i]]
            virtual_edge_index = torch.tensor(np.array(virtual_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)

            super_edge_index = [[i, num_atoms + virtual_num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(
                    self.edge_index_nosuper.device)

            self.edge_index = torch.cat((self.edge_index_nosuper, virtual_edge_index, super_edge_index), dim=1)


            virtual_edge_attr = torch.zeros(virtual_edge_index.size()[1], 2)


            virtual_edge_attr[:, 0] = 6  # bond type for self-loop edge
            virtual_edge_attr = virtual_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)

            super_edge_attr = torch.zeros(super_edge_index.size()[1], 2)

            super_edge_attr[:, 0] = 5  # bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            
            self.edge_attr = torch.cat((self.edge_attr_nosuper, virtual_edge_attr, super_edge_attr), dim=0)
            self.num_part = (num_atoms, virtual_num_atoms, 1)

        else:
            self.x = torch.cat((self.x_nosuper, super_x), dim=0)

            super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(
                self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)

            super_edge_attr = torch.zeros(num_atoms, 2)
            super_edge_attr[:, 0] = 5  # bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, super_edge_attr), dim=0)

            self.num_part = (num_atoms, 0, 1)

    def size_node(self):
        return self.x.size()[0]

    def size_edge(self):
        return self.edge_attr.size()[0]

    def size_atom(self):
        return self.x_nosuper.size()[0]

    def size_bond(self):
        return self.edge_attr_nosuper.size()[0]

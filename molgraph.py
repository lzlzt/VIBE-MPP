import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import math
from utils import * 




threshold = 10       
conf_nums = 20     



class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        with open(data_file) as f:
            self.data = [line.strip("\r\n ").split()[0] for line in f]
            # print('data',self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        # print('smiles',smiles)
        mol_graph = MolGraph(smiles) 
        # mol_tree.recover()
        # mol_tree.assemble()
        return mol_graph
    
class MyMol():
    def __init__(self,dic):
        self.x = dic['x']
        self.edge_index = dic['edge_index']
        self.edge_attr = dic['edge_attr']
        self.num_part = dic['num_part']
        self.size_ato = dic['size_atom']
        self.size_bon = dic['size_bond']
        self.size_edg = dic['size_edge']
        self.size_nod = dic['size_node']
        self.edge_index_nosuper = dic['edge_index_nosuper']
        self.edge_attr_nosuper = dic['edge_attr_nosuper']
        self.x_nosuper = dic['x_nosuper']
        self.distance = dic['distance']
    
    def size_node(self):
        return self.size_nod

    def size_edge(self):
        return self.size_edg

    def size_atom(self):
        return self.size_ato

    def size_bond(self):
        return self.size_bon
        
    

        
        
class MyMoleculeDataset(Dataset):

    def __init__(self, molgraphs):
        self.molgraphs = molgraphs
    def __len__(self):
        return len(self.molgraphs)

    def __getitem__(self, idx):
        mol_dic = self.molgraphs[idx]
        
        return mol_dic

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),  
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],   
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [                          
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,              
        Chem.rdchem.BondDir.ENDUPRIGHT,        
        Chem.rdchem.BondDir.ENDDOWNRIGHT      
    ],
    'possible_bond_inring': [None, False, True],
    'possible_bond_strength' :[0, 1,2,3,4,5,6,7,8,9,10, 200]
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

      
        self.x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        # num_bond_features = 2  # bond type, bond direction
        num_bond_features = 3  # bond type, bond direction, bond strength
        if len(self.mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in self.mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = [allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
                    bond.IsInRing())] + [200]

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

       
        num_atoms = self.x_nosuper.size(0) 

      
        super_x = torch.tensor([[119, 0]]).to(self.x_nosuper.device)
        virtual_mol = get_virtual_mol(self.mol)
        virtual_num_atoms = virtual_mol.GetNumAtoms()

        if virtual_num_atoms > 2:
            virtual_x = torch.tensor([[120, 0]]).repeat_interleave(virtual_num_atoms, dim=0).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, virtual_x, super_x), dim=0)

            virtual_edge_index = []
            strengths = []

            self.distances = get_distance(self.mol,self.conf)
           
            n = min(virtual_num_atoms,num_atoms)
            for i in range(n):
                for j in range(i + 1, n):
                    atom_i = self.mol.GetAtomWithIdx(i)
                    atom_j = self.mol.GetAtomWithIdx(j)
                    bond = self.mol.GetBondBetweenAtoms(i, j)
                    dist = self.distances[i, j]
                    if bond is not None:
                        pass
                    else:
                        if dist > threshold:
                            pass
                        else:
                            if not is_ring(self.mol,atom_i,atom_j):
                                virtual_edge_index = virtual_edge_index + [[i,num_atoms + j]] + [[j,num_atoms + i]]
                                strength = (10 - math.floor(dist))
                                strengths.append(strength)
                                strengths.append(strength)

            virtual_edge_index = torch.tensor(np.array(virtual_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)

            super_edge_index = [[i, num_atoms + virtual_num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(
                    self.edge_index_nosuper.device)

            self.edge_index = torch.cat((self.edge_index_nosuper, virtual_edge_index, super_edge_index), dim=1)


            virtual_edge_attr = torch.zeros(virtual_edge_index.size()[1], 3)

            virtual_edge_attr[:, 0] = 6  # bond type for self-loop edge
          
            virtual_edge_attr[:len(strengths), -1] = torch.tensor(strengths).to(virtual_edge_attr.device)
            virtual_edge_attr = virtual_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            
           
            super_edge_attr = torch.zeros(super_edge_index.size()[1], 3)

         
            super_edge_attr[:, 0] = 5  # bond type for self-loop edge
            super_edge_attr[:, 2] = 0 
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
    
    def molgraph_to_dict(molgraph):
        molgraph_dict = {}
        molgraph_dict['x'] = molgraph.x
        molgraph_dict['edge_index'] = molgraph.edge_index
        molgraph_dict['edge_attr'] = molgraph.edge_attr
        molgraph_dict['num_part'] = molgraph.num_part
        molgraph_dict['size_atom'] = molgraph.x_nosuper.size()[0]
        molgraph_dict['size_bond'] = molgraph.edge_attr_nosuper.size()[0]
        molgraph_dict['size_node'] = molgraph.x.size()[0]
        molgraph_dict['size_edge'] = molgraph.edge_attr.size()[0]
        molgraph_dict['edge_index_nosuper'] = molgraph.edge_index_nosuper
        molgraph_dict['edge_attr_nosuper'] = molgraph.edge_attr_nosuper
        molgraph_dict['x_nosuper'] = molgraph.x_nosuper
        molgraph_dict['distance'] = molgraph.distances
        return molgraph_dict
    
        def molgraph_from_dict(molgraph_dict):
            x = torch.tensor(molgraph_dict['x'])
            edge_index = torch.tensor(molgraph_dict['edge_index'])
            edge_attr = torch.tensor(molgraph_dict['edge_attr'])
            num_part = torch.tensor(molgraph_dict['num_part'])
            molgraph = MyMol(x, edge_index, edge_attr, num_part)
            return molgraph
    
    

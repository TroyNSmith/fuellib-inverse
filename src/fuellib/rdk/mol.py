"""RDKit Mol-related utilities and functions for the FuelLib package."""

from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol


# Instantiation functions
def from_smiles(smiles: str) -> Mol:
    """
    Instantiate an RDKit Mol object from a SMILES string.

    :param smiles: SMILES string representing the molecule.
    :type smiles: str
    :return: RDKit Mol object.
    :rtype: Mol
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol


def smiles(mol: Mol) -> str:
    """
    Get the SMILES string representation of an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: SMILES string representing the molecule.
    :rtype: str
    """
    return Chem.MolToSmiles(mol)


def from_inchi(inchi: str) -> Mol:
    """
    Instantiate an RDKit Mol object from an InChI string.

    :param inchi: InChI string representing the molecule.
    :type inchi: str
    :return: RDKit Mol object.
    :rtype: Mol
    """
    mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
    mol = Chem.AddHs(mol)
    return mol


def inchi(mol: Mol) -> str:
    """
    Get the standard InChI string representation of an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: InChI string representing the molecule.
    :rtype: str
    """
    return Chem.inchi.MolBlockToInchi(Chem.rdmolfiles.MolToMolBlock(mol))


# Structural feature checks
def is_hydrocarbon(mol: Mol) -> bool:
    """
    Check if an RDKit Mol object represents a hydrocarbon.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: True if the molecule contains only carbon and hydrogen atoms, False otherwise.
    :rtype: bool
    """
    return all(atom.GetSymbol() in {"C", "H"} for atom in mol.GetAtoms())


def has_branched(mol: Mol) -> bool:
    """
    Check if an RDKit Mol object contains any branched atoms.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: True if the molecule contains any branched atoms, False otherwise.
    :rtype: bool
    """
    mol = Chem.RemoveAllHs(mol)
    return any(atom.GetDegree() > 2 for atom in mol.GetAtoms())


def count_olefins(mol: Mol) -> int:
    """
    Count the number of olefins (non-aromatic double bonds) in an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: Number of olefins in the molecule.
    :rtype: int
    """
    count = 0
    for bond in mol.GetBonds():
        if (
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
            and not bond.GetIsAromatic()
        ):
            count += 1
    return count


def count_aliphatic_rings(mol: Mol) -> int:
    """
    Count the number of aliphatic (non-aromatic) rings in an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: Number of aliphatic rings in the molecule.
    :rtype: int
    """
    count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            count += 1
    return count


def count_aromatic_rings(mol: Mol) -> int:
    """
    Count the number of aromatic rings in an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: Number of aromatic rings in the molecule.
    :rtype: int
    """
    count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            count += 1
    return count


def atom_counts(mol: Mol) -> dict[str, int]:
    """
    Count the number of each type of atom in an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :return: Dictionary mapping atom symbols to their counts.
    :rtype: dict[str, int]
    """
    mol = Chem.AddHs(mol)
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms())


# Molecular property calculations
def molecular_weight(mol: Mol, *, exact: bool = False) -> float:
    """
    Calculate the molecular weight of an RDKit Mol object.

    :param mol: RDKit Mol object.
    :type mol: Mol
    :param exact: Whether to calculate the monoisotopic molecular weight.
    :type exact: bool
    :return: Molecular weight of the molecule in atomic mass units (amu).
    :rtype: float
    """
    if exact:
        return Descriptors.ExactMolWt(mol)  # ty: ignore[unresolved-attribute]
    return Descriptors.MolWt(mol)  # ty: ignore[unresolved-attribute]


__all__ = [
    "atom_counts",
    "count_aliphatic_rings",
    "count_aromatic_rings",
    "count_olefins",
    "molecular_weight",
]

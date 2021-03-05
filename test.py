from ase.io.vasp import read_vasp, write_vasp
from ase.io import read, write
from ase import Atoms
from ase.build import surface
from sagar.crystal.structure import Cell
# from sagar.io.vasp import read_vasp
from sagar.element.base import get_symbol
import numpy as np
from ase.build import diamond111
from pdb import set_trace


def cell2atoms(cell):
    pos = np.dot(cell.positions, cell.lattice)
    lat = cell.lattice
    num = cell.atoms
    atoms = Atoms(positions=pos, cell=lat, numbers=num)
    return atoms


def atoms2cell(atoms):
    pos = atoms.positions
    lat = atoms.cell
    num = atoms.numbers
    cell = Cell(positions=pos, lattice=lat, atoms=num)
    return cell


def ext_gcd(a, b):
    if b == 0:
        return 1, 0
    elif a % b == 0:
        return 0, 1
    else:
        x, y = ext_gcd(b, a % b)
        return y, x - y * (a // b)


def expend_supercell(poscar, size):
    atoms = read_vasp(poscar)
    a1, a2, a3 = atoms.cell
    c1,c2,c3 = [a1,a2,a3] * size
    pos = atoms.positions

    P = []
    for ii in range(size[0]):
        for jj in range(size[1]):
            for kk in range(size[2]):
                tmp = pos + np.dot([ii, jj, kk], atoms.cell)
                if len(P)==0:
                    P = tmp
                else:
                    P = np.vstack((P, tmp))
    num = np.tile(atoms.numbers, size[0] * size[1] * size[2])
    A = Atoms(positions=P, cell=[c1,c2,c3], numbers=num)
    # set_trace()
    write_vasp('POSCAR2.vasp', A, direct=True, sort=True, vasp5=True)


if __name__ == '__main__':
    # atoms = diamond111(symbol='C', size=(1, 1, 2), vacuum=10)
    # write_vasp('poscarslab.vasp', atoms, direct=True, sort=True, vasp5=True)
    atom = read_vasp('POSCAR1.vasp')
    slab = surface(atom, indices=(1, 1, 1), layers=1, vacuum=10)
    write_vasp('poscarSSlab.vasp', slab, direct=True, sort=True, vasp5=True)
    # poscar = 'POSCAR1.vasp'
    # size=np.array([2,2,2])
    # expend_supercell(poscar,size)
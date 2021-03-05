from ase.build import surface
from ase.io import read
from ase.io.vasp import write_vasp


def slice(poscar, indices, vacuum, layer=1, name=''):
    '''
    :param poscar: poscar'name
    :param indices: Miller indices
    :param vacuum: vacuum's length (A)
    :param layer: Number of equivalent layers of the slab
    :return: new poscar
    '''
    if name=='':
        name = '%s_slab%s' % (poscar, str(indices))
    else:
        pass
    POSCAR = poscar
    Atom = read(POSCAR)
    slab = surface(Atom, indices=indices, layers=layer, vacuum=vacuum)
    write_vasp(name, slab)


if __name__ == '__main__':
    slice('SiAu3_mp-972868_conventional_standard.vasp', (1, 1, 1), 12)

# Si3Au10_mp-1219349_conventional_standard.vasp
# SiAu3_mp-1186998_conventional_standard.vasp
# SiAu3_mp-972868_conventional_standard.vasp

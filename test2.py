from ase.build import fcc111, root_surface
from ase.io.vasp import write_vasp, read_vasp
from pdb import set_trace

atoms = fcc111('Ag', (1, 1, 3))
atoms = root_surface(atoms, 3)

set_trace()

write_vasp('poscarslab.vasp', atoms, direct=True, sort=True, vasp5=True)

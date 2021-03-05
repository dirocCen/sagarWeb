from ase import Atoms
from ase.build import surface
from pyvaspflow.utils import write_poscar, generate_all_basis,refine_points
# from pyvaspflow.utils import get_identity_atoms as pyvasp_get_identity_atoms
from sagar.crystal.structure import symbol2number as s2n
from sagar.crystal.structure import Cell
from sagar.element.base import get_symbol
from sagar.crystal.derive import ConfigurationGenerator,cells_nonredundant
from sagar.molecule.derive import ConfigurationGenerator as mole_CG
from sagar.toolkit.mathtool import refine_positions
from sagar.molecule.structure import Molecule
from sagar.io.vasp import _write_string
from ase.db import connect
from ase.io import read,write
from ase.io.vasp import write_vasp
from itertools import chain,combinations
import seekpath,os,shutil,time,zipfile
import numpy as np
import matplotlib.pyplot as plt
import operator as op
from functools import reduce
import os

# root = "/home/hecc/webapp/webapp"
root = "/home/ceny/MyDisk/study_document/sagarWeb"

def get_type_from_submit(form):
    for key,val in form.items():
        if val == "submit":
            return key

def get_perms(cell,str_type='crystal',symprec=1e-3):
    latt = cell.lattice
    pos = cell.positions
    pos = np.dot(pos,latt)
    if str_type == "crystal":
        symm = cell.get_symmetry()
        trans,rots = symm['translations'],symm['rotations']
        perms = np.zeros((np.shape(trans)[0],len(cell.atoms)))
        origin_positions = refine_positions(cell.positions)
        for ix, rot in enumerate(rots):
            for iy,o_pos in enumerate(origin_positions):
                new_pos = np.dot(rot,o_pos.T) + trans[ix]
                new_pos = np.mod(new_pos,1)
                new_pos = refine_positions(new_pos)
                idx = np.argmin(np.linalg.norm(new_pos-origin_positions,axis=1))
                perms[ix,iy] = idx
        perms_table = np.unique(perms,axis=0)
    else:
        mol = Molecule(pos,cell.atoms)
        perms_table = mol.get_symmetry_permutation(symprec)
    return perms_table

def automatic_linemode(structure,num_kpts=16):
    all_kpath = seekpath.get_path((structure.lattice,
    structure.positions,structure.atoms))
    pri_pos,pri_lat = all_kpath['primitive_positions'],all_kpath['primitive_lattice']
    res = "Note that the lattice maybe change during seeking the k-path <br>"
    res += "primitive_lattice: <br>"
    for lat in pri_lat:
        res += " ".join(["{:.7f}".format(i) for i in lat.tolist()]) + "<br>"
    res += "<br> <br> primitive_positions:<br>"

    for pos in pri_pos:
        res += " ".join(["{:.7f}".format(i) for i in pos.tolist()]) + "<br>"
    res += "<br> <br>"

    points = all_kpath['point_coords']
    path = all_kpath['path']
    kpoints,labels = [],[]
    for p in path:
        kpoints.append(points[p[0]])
        kpoints.append(points[p[1]])
        labels.append(p[0])
        labels.append(p[1])
    k_path = ""
    k_path += 'Line_mode KPOINTS file, '+'num_kpts: '+str(num_kpts) + "<br>"
    k_path += str(num_kpts) +"<br > Line_mode <br> Reciprocal <br >"
    for idx in range(0,len(kpoints),2):
        kpt = [str(k) for k in kpoints[idx]]
        k_path += "  ".join(kpt)
        if labels[idx] == "GAMMA":
            labels[idx] = "\Gamma"
        k_path += "  ! "+labels[idx] + "<br >"
        kpt = [str(k) for k in kpoints[idx+1]]
        k_path += "   ".join(kpt)
        if labels[idx+1] == "GAMMA":
            labels[idx+1] = "\Gamma"
        k_path += " !  "+labels[idx+1] + "<br ><br>"
    return res + k_path

def get_primitve_cell(cell,symprec):
    cell = cell.get_primitive_cell(symprec)
    cell_str =  _write_string(cell, long_format=True)
    return cell_str.replace("\n","<br >")

def get_niggli_cell(cell,dim):
    if dim == 2:
        cell = cell._get_niggli_2D()
    else:
        cenn = cell._get_niggli_3D()
    cell_str =  _write_string(cell, long_format=True)
    return cell_str.replace("\n","<br >")

def get_identity_atoms(cell,symprec,style="crystal"):
    atom_number = cell.atoms
    if style == "crystal":
        equ_atom = cell.get_symmetry(symprec)['equivalent_atoms']
        atom_uniq_type = np.unique(equ_atom)
        atom_type = np.zeros(np.shape(equ_atom))
        for idx,ea in enumerate(equ_atom):
            atom_type[idx] = np.where(atom_uniq_type==ea)[0]
    else:
        perms = get_perms(cell,str_type="molecule",symprec=symprec)
        n = np.shape(perms)[1]
        atom_type = np.zeros((n,1))
        tmp = np.unique(perms[:,np.unique(perms[:,0])])
        has_sorted = np.unique(np.hstack((np.zeros((1,)),tmp))).astype("int")
        sort_idx = 0
        atom_type[has_sorted] = sort_idx
        sort_idx += 1
        while True:
            unsorted = np.setdiff1d(range(n),has_sorted)
            tmp = np.unique(perms[:,np.unique(perms[:,unsorted[0]])])
            atom_type[tmp] = sort_idx
            sort_idx += 1
            has_sorted = np.unique(np.hstack((has_sorted,tmp))).astype("int")
            if len(has_sorted) == n:
                break
    res = "Lattice : <br >"
    for i in range(3):
        res += "   ".join(["%.7f"%(i) for i in cell.lattice[i].tolist()]) + "<br >"
    res += "positions: <br >"
    for idx,pos in enumerate(cell.positions):
        res += "   ".join(["%.7f"%(i) for i in pos.tolist()]) + " " \
        + get_symbol(atom_number[idx])\
        + " " +"%d"%(idx) \
        + " " +"{%d}"%(atom_type[idx]) +"<br >"
    return res

def get_point_defect(cell,symprec=1e-3,doped_out='all',doped_in=['Vac'],num=[1],ip=""):
    cg = ConfigurationGenerator(cell, symprec)
    sites = _get_sites(list(cell.atoms), doped_out=doped_out, doped_in=doped_in)
    if num == None:
        confs = cg.cons_specific_cell(sites, e_num=None, symprec=symprec)
        comment = ["-".join(doped_in)+"-all_concentration"]
    else:
        purity_atom_num = sum([1 if len(site)>1 else 0 for site in sites])
        confs = cg.cons_specific_cell(sites, e_num=[purity_atom_num-sum(num)]+num, symprec=symprec)
        comment = list(chain(*zip(doped_in, [str(i) for i in num])))
    comment = '-'.join(doped_out) +'-'+'-'.join(comment) + '-defect-' + str(int(time.time()))
    folder = os.path.join(root,"downloads",ip,comment)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    deg = []
    idx = 0
    for c, _deg in confs:
        write_poscar(c,folder,idx)
        deg.append(_deg)
        idx += 1
    np.savetxt(os.path.join(folder,"deg.txt"),deg,fmt='%d')
    zip_file = os.path.join(root,"downloads",ip,comment)
    zip_dir(folder,zip_file+'.zip')
    shutil.rmtree(folder)
    return os.path.join(root,"downloads",ip,zip_file+".zip")

def get_mole_point_defect(cell,symprec=1e-3,doped_out='all',doped_in=['Vac'],num=[1],ip=""):
    pos,lat,atoms = cell.positions,cell.lattice,cell.atoms
    mole = Molecule(np.dot(pos,lat),atoms)
    cg = mole_CG(mole, symprec)
    sites = _get_sites(list(mole.atoms), doped_out=doped_out, doped_in=doped_in)
    if num == None:
        confs = cg.get_configurations(sites, e_num=None)
        comment = ["-".join(doped_in)+"-all_concentration"]
    else:
        purity_atom_num = sum([1 if len(site)>1 else 0 for site in sites])
        confs = cg.get_configurations(sites, e_num=[purity_atom_num-sum(num)]+num)
        comment = list(chain(*zip(doped_in, [str(i) for i in num])))
    comment = '-'.join(doped_out) +'-'+'-'.join(comment) + '-defect-' + str(int(time.time()))
    folder = os.path.join(root,"downloads",ip,comment)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    deg = []
    idx = 0
    for c, _deg in confs:
        c.lattice = lat
        c._positions = np.dot(c.positions,np.linalg.inv(lat))
        write_poscar(c,folder,idx)
        deg.append(_deg)
        idx += 1
    np.savetxt(os.path.join(folder,"deg.txt"),deg,fmt='%d')
    zip_file = os.path.join(root,"downloads",ip,comment)
    zip_dir(folder,zip_file+'.zip')
    shutil.rmtree(folder)
    return os.path.join(root,"downloads",ip,zip_file+".zip")

def _get_sites(atoms,doped_out,doped_in):
    doped_in = [s2n(i) for i in doped_in]
    if doped_out == ['all']:
        return  [tuple([atom]+doped_in)  for atom in atoms]
    elif len(doped_out) > 1:
        doped_out = [s2n(i) for i in doped_out]
        return [tuple([atom]+doped_in) if atom in doped_out else (atom,) for atom in atoms ]
    else:
        doped_out = [s2n(i) for i in doped_out]
        _ins = tuple(doped_out+doped_in)
        return [_ins if atom == doped_out else (atom,) for atom in atoms]

def extend_specific_volume(pcell, dimension, volume, symprec=1e-3, comprec=1e-3,ip=""):
    (min_v, max_v) = volume
    comment = 'extend_cell-'+str(int(time.time()))
    folder = os.path.join(root,"downloads",ip,comment)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if min_v == -1:
        _export_supercell(pcell, comment, dimension, max_v, symprec, comprec,folder)
    else:
        for v in range(min_v, max_v + 1):
            _export_supercell(pcell, comment, dimension, v, symprec, comprec,folder)
    zip_file = os.path.join(root,"downloads",ip,comment)
    zip_dir(folder,zip_file+".zip")
    shutil.rmtree(folder)
    return zip_file+".zip"

def _export_supercell(pcell, comment, dimension, v, symprec, comprec, folder):
    cells = cells_nonredundant(pcell, v, dimension, symprec=symprec, comprec=comprec)
    for idx, c in enumerate(cells):
        write_poscar(c,folder=folder,idx=idx,comment="-v-"+str(v)+"-")

def spec_vol_point_defect(cell,doped_out,doped_in,volume,dimension,e_num,symprec,ip):
    sites = _get_sites(list(cell.atoms), doped_out=doped_out, doped_in=doped_in)
    cg = ConfigurationGenerator(cell, symprec)
    sites = _get_sites(list(cell.atoms), doped_out=doped_out, doped_in=doped_in)
    if e_num == None:
        confs = cg.cons_specific_volume(sites=sites, volume=volume, e_num=None, dimension=dimension, symprec=symprec)
        comment = ["-".join(doped_in)+"-all_concentration"]
    else:
        purity_atom_num = np.where(cell.atoms==s2n(doped_out))[0].size*volume
        if len(e_num) == 1:
            e_num = e_num[0]
            confs = cg.cons_specific_volume(sites=sites, volume=volume, e_num=(purity_atom_num-e_num, e_num), dimension=dimension, symprec=symprec)
            comment = list(chain(*zip(doped_in, [str(i) for i in [e_num] ])))
        else:
            _e_num = (purity_atom_num-sum(e_num),)+tuple(e_num)
            confs = cg.cons_specific_volume(sites=sites, volume=volume, e_num=_e_num, dimension=dimension, symprec=symprec)
            comment = list(chain(*zip(doped_in, [str(i) for i in e_num ])))

    comment = "".join(doped_out) +'-'+'-'.join(comment) + '-defect-v-'+str(volume)+"-"+ str(int(time.time()))
    folder = os.path.join(root,"downloads",ip,comment)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    deg = []
    idx = 0
    for c, _deg in confs:
        write_poscar(c,folder,idx)
        deg.append(_deg)
        idx += 1
    np.savetxt(os.path.join(folder,"deg.txt"),deg,fmt='%d')
    zip_file = os.path.join(root,"downloads",ip,comment)
    zip_dir(folder,zip_file+'.zip')
    shutil.rmtree(folder)
    return os.path.join(root,"downloads",ip,zip_file+".zip")

def zip_dir(dirpath,outFullName):
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        fpath = path.replace('/'.join(dirpath.split("/")[:-1]),'')
        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def transform_db(db_file):
    if not db_file.endswith(".db"):
        raise ValueError(db_file+" is not supported here")
    filename = db_file.split('.')[-2].split('/')[-1]
    if os.path.isdir(os.path.join("project",filename)):
        shutil.rmtree(os.path.join("project",filename))
    os.makedirs(os.path.join("project",filename))
    ats = read(db_file,':')
    for idx,at in enumerate(ats):
        write(os.path.join("project",filename,str(idx+1)+".cif"),at)
        print(idx)

def make_db(db_file):
    root_dir = "/home/hecc/Documents/vasp_data/C20H2"
    db = connect(db_file)
    for idx in range(1,6):
        at = read(os.path.join(root_dir,"scf"+str(idx),"OUTCAR"))
        db.write(at,key_value_pairs={"magnetic_moment":at.get_magnetic_moment()})

def write_png(filename,at,gap=None):
    pos = at.positions
    n = pos.shape[0]
    fig,ax = plt.subplots(1,1)
    for x,y,z in pos:
        ax.plot(x,y,'bo')
    for ii in range(n):
        for jj in range(ii+1,n):
            d = np.linalg.norm(pos[ii]-pos[jj])
            if abs(d-1.7)<0.2:
                ax.plot([pos[ii,0],pos[jj,0]],[pos[ii,1],pos[jj,1]],'b')
    if gap:
        ax.set_title("gap: "+"%.3f"%gap+" eV")
    ax.set_aspect('equal', 'box')
    plt.savefig(filename)
    plt.close()

def get_tetrahedral_defect(cell, isunique=False, doped_in='H',min_d=1.5,folder='tetrahedral-defect',ip=""):
    all_basis = generate_all_basis(1,1,1)
    direct_lattice = np.array([[1,0,0],[0,1,0],[0,0,1]])
    extend_S = np.zeros((0,3))
    for basis in all_basis:
        new_basis = np.sum([(direct_lattice[ii]*basis[ii]).tolist() for ii in range(3)],axis=0)
        extend_S = np.vstack((extend_S,
        cell.positions+np.tile(new_basis,len(cell.atoms)).reshape((-1,3))))
    idx = np.sum((extend_S <= 1.2) &(extend_S >= -0.2),axis=1)
    idx = np.where(idx == 3)[0]
    extend_S = np.dot(extend_S[idx],cell.lattice)
    n = extend_S.shape[0]
    d = np.zeros((n,n))
    for ii in range(n):
        d[ii,ii+1:] = np.linalg.norm(extend_S[ii]-extend_S[ii+1:],axis=1)
    d = d + d.T
    first_tetra,sec_tetra,third_tetra = [],[],[]
    for ii in range(n):
        temp_d = sorted(d[ii])
        idx = np.where(abs(d[ii] - temp_d[1])<min_d)[0]
        if len(idx) < 3:
            continue
        for comb in combinations(idx,3):
            comb_list = list(comb)
            tmp = d[comb_list][:,comb_list]
            comb_list.append(ii)
            if np.std(tmp[tmp>0]) < 0.01:
                if abs(tmp[0,1]-temp_d[1]) < 0.1:
                    first_tetra.append(comb_list)
                else:
                    sec_tetra.append(comb_list)
            else:
                tmp = d[comb_list][:,comb_list]
                tmp = np.triu(tmp)
                tmp = sorted(tmp[tmp>0])
                if (np.std(tmp[0:4]) < 0.1 or np.std(tmp[1:5]) < 0.1 or
                np.std(tmp[2:]) < 0.1) and np.std(tmp) < 0.5:
                    third_tetra.append(comb_list)
    all_tetra = []
    if len(first_tetra) != 0:
        first_tetra = np.unique(np.sort(first_tetra,axis=1),axis=0)
        first_tetra = refine_points(first_tetra,extend_S,cell.lattice,min_d=min_d)
        all_tetra.append(first_tetra)
    if len(sec_tetra) !=0:
        sec_tetra = np.unique(np.sort(sec_tetra,axis=1),axis=0)
        sec_tetra = refine_points(sec_tetra,extend_S,cell.lattice,min_d=min_d)
        all_tetra.append(sec_tetra)
    if len(third_tetra) != 0:
        third_tetra = np.unique(np.sort(third_tetra,axis=1),axis=0)
        third_tetra = refine_points(third_tetra,extend_S,cell.lattice,min_d=min_d)
        all_tetra.append(third_tetra)

    folder = os.path.join(root,"downloads",ip,folder)
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        shutil.rmtree(folder)
        os.mkdir(folder)
    idx = 0
    for tetra in all_tetra:
        if len(tetra) == 0:
            continue
        new_pos = np.vstack((cell.positions,tetra))
        new_atoms = np.hstack((cell.atoms,s2n(doped_in)*np.ones((tetra.shape[0],))))
        new_cell = Cell(cell.lattice,new_pos,new_atoms)
        write_poscar(new_cell,folder,idx)
        idx += 1
    zip_file = os.path.join(root,"downloads",ip,folder)
    zip_dir(folder,zip_file+'.zip')
    shutil.rmtree(folder)
    return os.path.join(root,"downloads",ip,zip_file+".zip")

def get_magnetic_config(cell,magnetic_atom,magmon=1,only_AFM=False,magmon_identity=False,symprec=1e-3,ip=""):
    cg = ConfigurationGenerator(cell, symprec)
    doped_in = get_symbol(np.setdiff1d(range(1,56),np.unique(cell.atoms))[0])
    sites = _get_sites(list(cell.atoms), doped_out=magnetic_atom, doped_in=[doped_in])
    n = len(cell.atoms)
    magmom = []
    magnetic_atom_idx = np.where(cell.atoms==s2n(magnetic_atom[0]))[0].astype("int")
    atoms_type = pyvasp_get_identity_atoms(cell,symprec)
    unique_atoms_type = np.unique(atoms_type)
    # import pdb; pdb.set_trace()
    if only_AFM:
        num_list = [len(magnetic_atom_idx)//2]
    else:
        num_list = range(len(magnetic_atom_idx)//2+1)
    for num in num_list:
        confs = cg.cons_specific_cell(sites, e_num=[len(magnetic_atom_idx)-num,num], symprec=symprec)
        for c,_def in confs:
            tmp_magmom = np.zeros((n,),dtype=int)
            tmp_magmom[magnetic_atom_idx] = magmon
            # import pdb; pdb.set_trace()
            mag_idx = [np.where(np.linalg.norm(cell.positions-refine_positions(c.positions)[_idx],axis=1)<0.01)[0][0] \
            for _idx in np.where(c.atoms==s2n(doped_in[0]))[0]]
            tmp_magmom[mag_idx] = -magmon
            if magmon_identity:
                flag = True
                for i in unique_atoms_type:
                    if len(np.unique(tmp_magmom[np.where(atoms_type==i)[0]])) != 1:
                        flag = False
                        break
                if flag:
                    magmom.append(tmp_magmom.tolist())
            else:
                magmom.append(tmp_magmom.tolist())
    # remove the equivalent AFM -1 1/1 -1
    final_magmom = set()
    final_magmom.add(tuple(magmom[0]))
    for idx in range(1,len(magmom)):
        mag = np.array(magmom[idx]).astype("int")
        if tuple(mag) in final_magmom:
            continue
        idx_up = np.where(mag==magmon)[0].astype("int")
        idx_down = np.where(mag==-magmon)[0].astype("int")
        mag[idx_up] = -magmon
        mag[idx_down] = magmon
        if tuple(mag) in final_magmom:
            continue
        final_magmom.add(tuple(mag))
    final_magmom = np.array([list(i) for i in final_magmom])
    incar_file = os.path.join(root,"downloads",ip,"INCAR-magmon")
    np.savetxt(incar_file,magmom,fmt='%2d')
    return incar_file

def cell2atoms(cell):
    pos = cell.positions @ cell.lattice
    lat = cell.lattice
    num = cell.atoms
    atoms = Atoms(positions=pos, cell=lat, numbers=num)
    return atoms

def get_slice_surface(atom, indice, layer, vacuum, name, ip):
    comment = 'slice_surface-' + str(int(time.time()))
    folder = os.path.join(root, "downloads", ip, comment)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    if len(layer) == 1:
        pass
    elif len(layer) == 2:
        layer = range(layer[0], layer[1] + 1)
    else:
        print('your input format error')
    for ii, L in enumerate(layer):
        slab = surface(atom, indices=indice, layers=L, vacuum=vacuum)
        write_vasp(os.path.join(folder, name + str(ii)), slab, direct=True, sort=True, vasp5=True)
    zip_dir(folder, folder + '.zip')
    shutil.rmtree(folder)
    return folder + ".zip"


if __name__ == '__main__':
    transform_db('db/In2O3-Mg.db')

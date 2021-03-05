#!/home/hecc/webapp/venv/bin/python3
from flask import Flask, render_template, flash, redirect, url_for, request,send_file,flash,send_from_directory
from sagar.crystal.structure import periodic_table_dict as ptd
from sagar.crystal.structure import symbol2number as s2n
from sagar.io.vasp import _read_string
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, render_template, request, session, jsonify
from ase.io import write
from utils import zip_dir,write_png, cell2atoms
import urllib.request
import os,utils,shutil,hashlib,json\
    # ,httpagentparser
import numpy as np
import collections
import functools
import io,random,string
import os.path as op
import re
import sys
import tempfile
# try:
#     import matplotlib
#     matplotlib.use('Agg', warn=False)
# except ImportError:
#     pass
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import ase.db
import ase.db.web
from ase.db.core import convert_str_to_int_float_or_str
# from ase.db.plot import atoms2png
# from ase.db.summary import Summary
from ase.db.table import Table, all_columns
from ase.visualize import view
from ase import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from plotly.offline import plot as pplot
import plotly.graph_objs as go
import re
from ase.build import surface
from ase.io import read
from ase.io.vasp import write_vasp

app = Flask(__name__)

limiter = Limiter(
    app,
    key_func=get_remote_address
)

root = "/home/ceny/MyDisk/study_document/sagarWeb"
# root = "/home/hecc/webapp/webapp"

app.secret_key = os.urandom(24)


@app.route("/", methods=['GET','POST'])
def indx():
    # return ipinfo.browser+ipinfo.ipaddress+ipinfo.os+ipinfo.lang
    return render_template("index.html")


@app.route('/sagar', methods=['GET', 'POST'])
def sagar():
    return render_template('sagar.html')


@app.route('/admin', methods=['GET', "POST"])
def admin():
    ip = request.remote_addr
    return render_template('admin.html',ip=ip)


@limiter.limit("1/5seconds")
@app.route('/sagar/result', methods=['POST'])
def result():
    ip = request.remote_addr
    form = request.form
    downloads = os.path.join(root,"downloads",ip)
    if not os.path.isdir(downloads):
        os.makedirs(downloads)
    # form = dict((key, request.form.getlist(key)) for key in request.form.keys())
    poscar = form.get("POSCAR")
    try:
        cell = _read_string(poscar)
    except:
        return "POSCAR format error, please check carefully your POSCAR"
    key = utils.get_type_from_submit(form)
    if key == "space_group":
        symprec = float(form.get("symprec_space_group"))
        spg = cell.get_spacegroup(symprec)
        return spg
    elif key == "niggli":
        dim = int(form.get("dim_nigg"))
        return utils.get_niggli_cell(cell,dim)
    elif key == "perm_table":
        symprec,stru_type = float(form.get("symprec_perm_table")),form.get("stru_type")
        perms = utils.get_perms(cell=cell,str_type=stru_type,symprec=symprec)+1
        perms_file = os.path.join(root,"downloads",ip,"perms.txt")
        np.savetxt(perms_file,perms,fmt="%d")
        return send_file(perms_file, as_attachment=True)
    elif key == "k_path":
        res = utils.automatic_linemode(structure=cell)
        return res
    elif key == "primitive_cell":
        symprec = float(form.get("symprec_primitive_cell"))
        res = utils.get_primitve_cell(cell=cell,symprec=symprec)
        return res
    elif key == "Identity_atoms":
        symprec = float(form.get("symprec_identity_atoms"))
        style = form.get("stru_type_identity")
        res = utils.get_identity_atoms(cell,symprec,style=style)
        return res
    elif key == "tetra_site":
        min_dis = form.get("min_dis")
        ele_ins = form.get("ele_ins_tetra")
        if ele_ins == "":
            return "please input 'Element Inserted' "
        try:
            min_dis = float(min_dis)
            if min_dis <= 0:
                return "A float number that is greater than zero is needed here"
        except:
            return "minimum distance %s cannot be converted to float"%(min_dis)
        return send_file(utils.get_tetrahedral_defect(cell, isunique=False, doped_in=ele_ins
        ,min_d=min_dis,folder='tetrahedral-defect',ip=ip),as_attachment=True)
    elif key == "mag_config":
        magnetic_atom = form.get("magnetic_atom")
        magmon = form.get("spin_num")
        only_AFM = form.get("only_AFM")
        magmon_identity = form.get("magmon_identity")
        if magnetic_atom == "":
            return "please input 'magnetic atom' "
        elif magnetic_atom not in ptd:
            return "%s is not in table of periodic elements"%magnetic_atom
        if magmon_identity.startswith("T"):
            magmon_identity = True
        else:
            magmon_identity = False
        if only_AFM.startswith("T"):
            only_AFM = True
            idx = len(np.where(cell.atoms==s2n(magnetic_atom))[0])
            if idx % 2 == 1:
                return "The number of magnetic atoms is %d, cannot generate AFM configurations"%idx
        else:
            only_AFM = False
        try:
            magmon = int(magmon)
        except:
            return "spin number %s cannot be converted to float"%(magmon)
        symprec = float(form.get("symprec_mag_conf"))
        return send_file(utils.get_magnetic_config(cell,[magnetic_atom],magmon=magmon,only_AFM=only_AFM,
        magmon_identity=magmon_identity,symprec=symprec,ip=ip),as_attachment=True)
    elif key == "spec_cell":
        symprec = float(form.get("symprec_spec_cell"))
        ele_ins,ele_rm,ele_num = form.get("ele_ins_spec_cell"),form.get("ele_rm_spec_cell"),form.get("num_spec_cell")
        if ele_ins == "":
            return "please input 'Element Inserted' "
        elif ele_rm == "":
            return "please input 'Element Removed' "
        elif ',' in ele_rm and ',' in ele_ins:
            return "multi-elements remove and insert is ambiguous, which cannot be supported."
        ele_ins = ele_ins.split(",")
        ele_ins = [ele.strip() for ele in ele_ins]
        for ele in ele_ins:
            if ele not in ptd:
                return ele + " is not in our periodic table,please check carefully or change another element"
        ele_rm = ele_rm.split(",")
        ele_rm = [ele.strip() for ele in ele_rm]
        for ele in ele_rm:
            if ele not in ptd and ele.lower() != 'all':
                return ele + " is not in our periodic table, please check carefully or change another element"
        if ele_num == "":
            ele_num = None
            _idx = 0
            for ele in ele_rm:
                _idx += len(np.where(cell.atoms==s2n(ele))[0])
            max_num = (1+len(ele_rm))**_idx

            if max_num > 1e5:
                return "We cannot generate  too many configurations with this little memory,\
                <br> you can fund us, which will also efficiently help your work. :)"
        else:
            ele_num = ele_num.split(",")
            ele_num = [int(i) for i in ele_num]
            if len(ele_num) != len(ele_ins):
                return "please confirm the number of element inserted and # of inserted number are consistent"
            max_num = 1
            if len(ele_rm) == 1:
                n = len(np.where(cell.atoms==s2n(ele_rm[0]))[0])
                for idx in range(len(ele_ins)):
                    max_num *= utils.ncr(n,ele_num[idx])
            else:
                _idx = 0
                for ele in ele_rm:
                    _idx += len(np.where(cell.atoms==s2n(ele))[0])
                max_num = utils.ncr(_idx,ele_num[0])
            if max_num > 1e5:
                return "We cannot generate  too many configurations with this little memory,\
                <br> you can fund us, which will also efficiently help your work. :)"
        return send_file(utils.get_point_defect(cell,symprec=symprec,doped_out=ele_rm,doped_in=ele_ins,num=ele_num,ip=ip),as_attachment=True)
    elif key == "extend_cell":
        if not cell.is_primitive():
            return "Please input a prmitive cell POSCAR"
        else:
            dim,vol = int(form.get("dim_extend_cell")),form.get("num_vol_extend_cell")
            if vol == "":
                return "please input '# of Inserted element' "
            vol = vol.split(",")
            if len(vol) == 1:
                vol = [int(vol[0]),int(vol[0])]
            elif len(vol) == 2:
                vol = [int(vol[0]),int(vol[1])]
            else:
                return "Bad format for 'Specific volume' "
            ip = request.remote_addr
            return send_file(utils.extend_specific_volume(cell, dim, vol, symprec=1e-3, comprec=1e-3, ip=ip),as_attachment=True)
    elif key == "spec_vol":
        symprec = float(form.get("symprec_spec_vol"))
        if not cell.is_primitive():
            return "Please input a prmitive cell POSCAR <br> below is the primitive cell: <br>"+\
            utils.get_primitve_cell(cell=cell,symprec=symprec)
        else:
            dim,vol = int(form.get("dim_spec_vol")),form.get("num_vol_spec_vol")
            ele_ins,ele_rm = form.get("ele_ins_spec_vol"),form.get("ele_rm_spec_vol")
            if "," in ele_ins or "," in ele_rm:
                return "only substition of one element is supported here <br> "
            if ele_ins == "":
                return "please input 'Element Inserted' "
            elif ele_rm == "":
                return "please input 'Element Removed' "
            elif vol == "":
                return "please input 'Specific volume' "
            # if ele_num == "":
            ele_num = None
            # else:
                # ele_num = [int(i) for i in ele_num.split(",")]
            ip = request.remote_addr
            return send_file(utils.spec_vol_point_defect(cell,doped_out=[ele_rm],doped_in=[ele_ins],volume=int(vol),
            dimension=dim,e_num=ele_num,symprec=symprec,ip=ip),as_attachment=True)
    elif key == "spec_mole":
        symprec = float(form.get("symprec_spec_mole"))
        ele_ins,ele_rm,ele_num = form.get("ele_ins_spec_mole"),form.get("ele_rm_spec_mole"),form.get("num_spec_mole")
        if ele_ins == "":
            return "please input 'Element Inserted' "
        elif ele_rm == "":
            return "please input 'Element Removed' "
        elif ',' in ele_rm and ',' in ele_ins:
            return "multi-elements remove and insert is ambiguous, which cannot be supported."
        ele_ins = ele_ins.split(",")
        ele_ins = [ele.strip() for ele in ele_ins]
        for ele in ele_ins:
            if ele not in ptd:
                return ele + " is not in our periodic table,please check carefully or change another element"
        ele_rm = ele_rm.split(",")
        ele_rm = [ele.strip() for ele in ele_rm]
        for ele in ele_rm:
            if ele not in ptd and ele.lower() != 'all':
                return ele + " is not in our periodic table, please check carefully or change another element"
        if ele_num == "":
            ele_num = None
            _idx = 0
            for ele in ele_rm:
                _idx += len(np.where(cell.atoms==s2n(ele))[0])
            max_num = (1+len(ele_rm))**_idx

            if max_num > 1e5:
                return "We cannot generate  too many configurations with this little memory,\
                <br> you can fund us, which will also efficiently help your work. :)"
        else:
            ele_num = ele_num.split(",")
            ele_num = [int(i) for i in ele_num]
            if len(ele_num) != len(ele_ins):
                return "please confirm the number of element inserted and # of inserted number are consistent"
            max_num = 1
            if len(ele_rm) == 1:
                n = len(np.where(cell.atoms==s2n(ele_rm[0]))[0])
                for idx in range(len(ele_ins)):
                    max_num *= utils.ncr(n,ele_num[idx])
            else:
                _idx = 0
                for ele in ele_rm:
                    _idx += len(np.where(cell.atoms==s2n(ele))[0])
                max_num = utils.ncr(_idx, ele_num[0])
            if max_num > 1e5:
                return "We cannot generate  too many configurations with this little memory,\
                <br> you can fund us, which will also efficiently help your work. :)"
        return send_file(utils.get_mole_point_defect(cell,symprec=symprec,doped_out=ele_rm,doped_in=ele_ins,num=ele_num,ip=ip),as_attachment=True)
    elif key == 'slice_surface':
        indice = tuple(map(int, form.get("indice").split(',')))
        vacuum = float(form.get("vacuum"))
        layer = list(map(int, form.get("layer").split(',')))
        name = 'POSCAR'
        atom = cell2atoms(cell)
        return send_file(utils.get_slice_surface(atom, indice, layer, vacuum, name, ip), as_attachment=True)




#
# # Every client-connetions gets one of these tuples:
# Connection = collections.namedtuple(
#     'Connection',
#     ['query',  # query string
#      'nrows',  # number of rows matched
#      'page',  # page number
#      'columns',  # what columns to show
#      'sort',  # what column to sort after
#      'limit'])  # number of rows per page
#
# app.secret_key = 'aasdasrrewg98354n;hgf]sdf'
#
# app.config.update(SEND_FILE_MAX_AGE_DEFAULT=0)
#
# databases = {}  # Dict[str, Database]
# open_ase_gui = False  # click image to open ASE's GUI
# download_button = True
#
#     # List of (project-name, title, nrows) tuples (will be filled in at run-time):
# projects = []  # List[Tuple[str, str, int]]
#
#     # Find numbers in formulas so that we can convert H2O to H<sub>2</sub>O:
# SUBSCRIPT = re.compile(r'(\d+)')
#
# next_con_id = 1
# connections = {}
# python_configs = []
# dbs = []
#
# uris = [root+'/db/'+i for i in os.listdir(root+'/db')]
#
#
#
# for uri in uris:
#     if uri.endswith('.py'):
#         python_configs.append(uri)
#         continue
#     if uri.startswith('postgresql://'):
#         project = uri.rsplit('/', 1)[1]
#     else:
#         project = uri.rsplit('/', 1)[-1].split('.')[0]
#     db = ase.db.connect(uri)
#     db.python = None
#     databases[project] = db
#     dbs.append(db)
#
# for py, db in zip(python_configs, dbs):
#     db.python = py
#
#
#
# """Initialize databases and fill in projects list."""
# for proj, db in sorted(databases.items()):
#     meta = ase.db.web.process_metadata(db)
#     db.meta = meta
#     nrows = len(db)
#     projects.append((proj, db.meta.get('title', proj), nrows))
#     print('Initialized {proj}: (rows: {nrows})'
#           .format(proj=proj, nrows=nrows))
#
#
#
#
# if 'ASE_DB_APP_CONFIG' in os.environ:
#     app.config.from_envvar('ASE_DB_APP_CONFIG')
#     path = app.config['ASE_DB_TEMPLATES']
#     app.jinja_loader.searchpath.insert(0, str(path))
#     connect_databases(str(name) for name in app.config['ASE_DB_NAMES'])
#     initialize_databases()
#     tmpdir = str(app.config['ASE_DB_TMPDIR'])
#     download_button = app.config['ASE_DB_DOWNLOAD']
#     # import pdb; pdb.set_trace()
#     open_ase_gui = False
# else:
#     tmpdir = tempfile.mkdtemp(prefix='ase-db-app-')  # used to cache png-files
#
#
#
# @app.route('/database', defaults={'project': None})
# @app.route('/database/<project>/')
# def index(project):
#     global next_con_id
#     # Backwards compatibility:
#     project = request.args.get('project') or project
#     # import pdb; pdb.set_trace()
#     if project is None and len(projects) > 1:
#         return render_template('projects.html',
#                                projects=projects,
#                                md=None)
#
#     if project is None:
#         project = list(databases)[0]
#
#     con_id = int(request.args.get('x', '0'))
#     if con_id in connections:
#         query, nrows, page, columns, sort, limit = connections[con_id]
#
#     if con_id not in connections:
#         # Give this connetion a new id:
#         con_id = next_con_id
#         next_con_id += 1
#         query = ['', {}, '']
#         nrows = None
#         page = 0
#         columns = None
#         sort = 'id'
#         limit = 25
#
#
#     db = databases.get(project)
#     if db is None:
#         return 'No such project: ' + project
#     meta = db.meta
#
#     if columns is None:
#         columns = meta.get('default_columns')[:] or list(all_columns)
#
#     if 'sort' in request.args:
#         column = request.args['sort']
#         if column == sort:
#             sort = '-' + column
#         elif '-' + column == sort:
#             sort = 'id'
#         else:
#             sort = column
#         page = 0
#     elif 'query' in request.args:
#         dct = {}
#         query = [request.args['query']]
#         q = query[0]
#         for special in meta['special_keys']:
#             kind, key = special[:2]
#             if kind == 'SELECT':
#                 value = request.args['select_' + key]
#                 dct[key] = convert_str_to_int_float_or_str(value)
#                 if value:
#                     q += ',{}={}'.format(key, value)
#             elif kind == 'BOOL':
#                 value = request.args['bool_' + key]
#                 dct[key] = convert_str_to_int_float_or_str(value)
#                 if value:
#                     q += ',{}={}'.format(key, value)
#             elif kind == 'RANGE':
#                 v1 = request.args['from_' + key]
#                 v2 = request.args['to_' + key]
#                 var = request.args['range_' + key]
#                 dct[key] = (v1, v2, var)
#                 if v1:
#                     q += ',{}>={}'.format(var, v1)
#                 if v2:
#                     q += ',{}<={}'.format(var, v2)
#             else:  # SRANGE
#                 v1 = request.args['from_' + key]
#                 v2 = request.args['to_' + key]
#                 dct[key] = (int(v1) if v1 else v1,
#                             int(v2) if v2 else v2)
#                 if v1:
#                     q += ',{}>={}'.format(key, v1)
#                 if v2:
#                     q += ',{}<={}'.format(key, v2)
#         q = q.lstrip(',')
#         query += [dct, q]
#         sort = 'id'
#         page = 0
#         nrows = None
#     elif 'limit' in request.args:
#         limit = int(request.args['limit'])
#         page = 0
#     elif 'page' in request.args:
#         page = int(request.args['page'])
#
#     if 'toggle' in request.args:
#         column = request.args['toggle']
#         if column == 'reset':
#             columns = meta.get('default_columns')[:] or list(all_columns)
#         else:
#             if column in columns:
#                 columns.remove(column)
#                 if column == sort.lstrip('-'):
#                     sort = 'id'
#                     page = 0
#             else:
#                 columns.append(column)
#
#     okquery = query
#
#     if nrows is None:
#         try:
#             nrows = db.count(query[2])
#         except (ValueError, KeyError) as e:
#             flash(', '.join(['Bad query'] + list(e.args)))
#             okquery = ('', {}, 'id=0')  # this will return no rows
#             nrows = 0
#
#     table = Table(db, meta.get('unique_key', 'id'))
#     table.select(okquery[2], columns, sort, limit, offset=page * limit)
#
#     con = Connection(query, nrows, page, columns, sort, limit)
#     connections[con_id] = con
#
#     if len(connections) > 1000:
#         # Forget old connections:
#         for cid in sorted(connections)[:200]:
#             del connections[cid]
#
#     table.format(SUBSCRIPT)
#
#     addcolumns = sorted(column for column in all_columns + table.keys
#                         if column not in table.columns)
#
#     return render_template('table.html',
#                            project=project,
#                            t=table,
#                            md=meta,
#                            con=con,
#                            x=con_id,
#                            pages=pages(page, nrows, limit),
#                            nrows=nrows,
#                            addcolumns=addcolumns,
#                            row1=page * limit + 1,
#                            row2=min((page + 1) * limit, nrows),
#                            download_button=download_button)
#
#
# @app.route('/database/<project>/image/<name>')
# def image(project, name):
#     id = int(name[:-4])
#     name = project + '-' + name
#     path = op.join(tmpdir, name)
#     if not op.isfile(path):
#         db = databases[project]
#         atoms = db.get_atoms(id)
#         atoms2png(atoms, path)
#
#     return send_from_directory(tmpdir, name)
#
#
# @app.route('/database/<project>/cif/<name>')
# def cif(project, name):
#     id = int(name[:-4])
#     name = project + '-' + name
#     path = op.join(tmpdir, name)
#     if not op.isfile(path):
#         db = databases[project]
#         atoms = db.get_atoms(id)
#         atoms.write(path)
#     return send_from_directory(tmpdir, name)
#
#
# @app.route('/database/<project>/plot/<uid>/<png>')
# def plot(project, uid, png):
#     png = project + '-' + uid + '-' + png
#     return send_from_directory(tmpdir, png)
#
#
# @app.route('/database/<project>/gui/<int:id>')
# def gui(project, id):
#     if open_ase_gui:
#         db = databases[project]
#         atoms = db.get_atoms(id)
#         view(atoms)
#     return '', 204, []
#
# @app.route('/<project>/cif/<int:id>.cif')
# def cif_file(project, id):
#     return send_file(root+'/project/'+project+'/'+str(id)+'.cif')
#
# @app.route('/database/<project>/dos/<int:id>')
# def dos(project, id):
#     dos_file = os.path.join("templates/",project,"dos","dos"+str(id)+".html")
#     if os.path.isfile(dos_file):
#         return render_template(project+"/dos/dos"+str(id)+".html")
#     return "DOS informations not found"
#
#
# @app.route('/database/<project>/band/<int:id>')
# def band(project, id):
#     band_file = os.path.join(root,"templates/",project,"band","band"+str(id)+".html")
#     if os.path.isfile(band_file):
#         return render_template(os.path.join(project,"band","band"+str(id)+".html"))
#     return "band informations() not found"
#
# @app.route('/database/<project>/row/<uid>')
# def row(project, uid):
#     db = databases[project]
#     if not hasattr(db, 'meta'):
#         db.meta = ase.db.web.process_metadata(db)
#     prefix = '{}/{}-{}-'.format(tmpdir, project, uid)
#     key = db.meta.get('unique_key', 'id')
#     try:
#         uid = int(uid)
#     except ValueError:
#         pass
#     row = db.get(**{key: uid})
#     s = Summary(row, db.meta, SUBSCRIPT, prefix)
#     atoms = Atoms(cell=row.cell, pbc=row.pbc)
#     n1, n2, n3 = kptdensity2monkhorstpack(atoms,
#                                           kptdensity=1.8,
#                                           even=False)
#     return render_template('summary.html',
#                            project=project,
#                            s=s,
#                            uid=uid,
#                            n1=n1,
#                            n2=n2,
#                            n3=n3,
#                            back=True,
#                            md=db.meta,
#                            open_ase_gui=open_ase_gui)
#
#
# def tofile(project, query, type, limit=0):
#     fd, name = tempfile.mkstemp(suffix='.' + type)
#     con = ase.db.connect(name, use_lock_file=False)
#     db = databases[project]
#     for row in db.select(query, limit=limit):
#         con.write(row,
#                   data=row.get('data', {}),
#                   **row.get('key_value_pairs', {}))
#     os.close(fd)
#     data = open(name, 'rb').read()
#     os.unlink(name)
#     return data
#
#
# def download(f):
#     @functools.wraps(f)
#     def ff(*args, **kwargs):
#         text, name = f(*args, **kwargs)
#         if name is None:
#             return text
#         headers = [('Content-Disposition',
#                     'attachment; filename="{}"'.format(name)),
#                    ]  # ('Content-type', 'application/sqlite3')]
#         return text, 200, headers
#     return ff
#
#
# @app.route('/database/<project>/xyz/<int:id>')
# @download
# def xyz(project, id):
#     fd = io.StringIO()
#     from ase.io.xyz import write_xyz
#     db = databases[project]
#     write_xyz(fd, db.get_atoms(id))
#     data = fd.getvalue()
#     return data, '{}.xyz'.format(id)
#
# if download_button:
#     @app.route('/database/<project>/vasp')
#     # @download
#     def vaspall(project):
#         ip = request.remote_addr
#         con_id = int(request.args['x'])
#         con = connections[con_id]
#         query=con.query[2]
#         # import pdb; pdb.set_trace()
#         limit=con.limit
#         tmpdir = os.path.join(root,'downloads',ip,'vasp')
#         if os.path.isdir(tmpdir):
#             shutil.rmtree(tmpdir)
#         os.makedirs(tmpdir)
#
#         db = databases[project]
#         for idx,row in enumerate(db.select(query)):
#             at = Atoms(cell=row.cell,positions=row.positions,symbols=row.symbols)
#             write(os.path.join(tmpdir,'POSCAR'+str(idx)),at,vasp5=True)
#         zip_dir(dirpath=tmpdir,outFullName=os.path.join(root,"downloads",ip,"selection_vasp.zip"))
#         shutil.rmtree(tmpdir)
#         return send_file(os.path.join(root,"downloads",ip,"selection_vasp.zip"), as_attachment=True)
#
#
# if download_button:
#     @app.route('/database/<project>/image')
#     # @download
#     def imageall(project):
#         ip = request.remote_addr
#         con_id = int(request.args['x'])
#         con = connections[con_id]
#         query=con.query[2]
#         # import pdb; pdb.set_trace()
#         limit=con.limit
#         tmpdir = os.path.join(root,'downloads',ip,'image')
#         if os.path.isdir(tmpdir):
#             shutil.rmtree(tmpdir)
#         os.makedirs(tmpdir)
#
#         db = databases[project]
#         for idx,row in enumerate(db.select(query)):
#             at = Atoms(cell=row.cell,positions=row.positions,symbols=row.symbols)
#             try:
#                 gap = row.gap
#             except:
#                 gap=None
#             write_png(os.path.join(tmpdir,str(idx)+".png"),at,gap=gap)
#         zip_dir(dirpath=tmpdir,outFullName=os.path.join(root,"downloads",ip,"selection_image.zip"))
#         shutil.rmtree(tmpdir)
#         return send_file(os.path.join(root,"downloads",ip,"selection_image.zip"), as_attachment=True)
#
# if download_button:
#     @app.route('/database/<project>/last_column')
#     # @download
#     def last_column(project):
#         ip = request.remote_addr
#         con_id = int(request.args['x'])
#         con = connections[con_id]
#         query=con.query[2]
#         # import pdb; pdb.set_trace()
#         limit=con.limit
#         tmp_file = os.path.join(root,'downloads',ip,con.columns[-1])
#         db = databases[project]
#         with open(tmp_file,"w") as f:
#             for idx,row in enumerate(db.select(query)):
#                 f.writelines("%s"%row.get(con.columns[-1])+"\n")
#
#         return send_file(tmp_file, as_attachment=True)
#
# @app.route("/database/<project>/plot_convex")
# def plot_convex(project):
#     #TODO: add row namedtuple to store these attributes
#     ip = request.remote_addr
#     con_id = int(request.args['x'])
#     con = connections[con_id]
#     query = con.query[2]
#     limit = con.limit
#     db = databases[project]
#
#     if len(con.columns) < 3:
#         return "Not enough columns"
#     tmpdir = os.path.join(root,'downloads',ip,'fig')
#     if os.path.isdir(tmpdir):
#         shutil.rmtree(tmpdir)
#     os.makedirs(tmpdir)
#     points = []
#     customdata = []
#     row_idx = []
#     for idx,row in enumerate(db.select(query)):
#         # at = Atoms(cell=row.cell,positions=row.positions,symbols=row.symbols)
#         # import pdb; pdb.set_trace()
#         try:
#             float(row.get(con.columns[-2]))
#         except:
#             return con.columns[-2]+ "can not be converted to number."
#
#         try:
#             float(row.get(con.columns[-1]))
#         except:
#             return con.columns[-1]+ "can not be converted to number."
#
#         points.append([float(row.get(con.columns[-2])), float(row.get(con.columns[-1]))])
#         # import pdb; pdb.set_trace()
#         customdata.append('http://hecc.vip:8080/database/'+project+'/row/'+str(row.id))
#         row_idx.append(row.id)
#     pts = np.array(points)
#
#
#     data = [
#         go.Scatter(
#             x=pts[:,0],
#             y=pts[:,1],
#             mode='markers',
#             marker=dict(
#                 size=14
#             ),
#             # name='mapbox 1',
#             # text=['Montreal'],
#             customdata=customdata
#         )
#     ]
#
#     hull = ConvexHull(pts)
#     hull_idx = []
#     customdata = []
#     for simplex in hull.simplices:
#         data.append(go.Scatter(
#             x=pts[simplex,0],
#             y=pts[simplex,1],
#             mode='lines+markers',
#             marker=dict(
#                 size=14,color="red"
#             ),
#             customdata = ['http://hecc.vip:8080/database/'+project+'/row/'+str(row_idx[simplex[0]]),
#             'http://hecc.vip:8080/database/'+project+'/row/'+str(row_idx[simplex[1]])]
#         ))
#
#
#     # Build layout
#     layout = go.Layout(
#         hovermode='closest',
#     )
#
#     # Build Figure
#     fig = go.Figure(
#         data=data,
#         # layout=layout,
#     )
#
#     fig.update_layout(showlegend=False,
#         xaxis_title=con.columns[-2],
#         yaxis_title=con.columns[-1])
#
#     # Get HTML representation of plotly.js and this figure
#     plot_div = pplot(fig, output_type='div', include_plotlyjs=True)
#     # Get id of html div element that looks like
#     # <div id="301d22ab-bfba-4621-8f5d-dc4fd855bb33" ... >
#     res = re.search('<div id="([^"]*)"', plot_div)
#     div_id = res.groups()[0]
#
#     # Build JavaScript callback for handling clicks
#     # and opening the URL in the trace's customdata
#     js_callback = """
#     <script>
#     var plot_element = document.getElementById("{div_id}");
#     plot_element.on('plotly_click', function(data){{
#         console.log(data);
#         var point = data.points[0];
#         if (point) {{
#             console.log(point.customdata);
#             window.open(point.customdata);
#         }}
#     }})
#     </script>
#     """.format(div_id=div_id)
#
#     # Build HTML string
#     html_str = """
#     <html>
#     <body>
#     {plot_div}
#     {js_callback}
#     </body>
#     </html>
#     """.format(plot_div=plot_div, js_callback=js_callback)
#
#     # Write out HTML file
#     return html_str
#
#
#
#
#
# @app.route('/database/<project>/vasp/<int:id>')
# @download
# def vasp(project, id):
#     fd = io.StringIO()
#     from ase.io.vasp import write_vasp
#     db = databases[project]
#     write_vasp(fd, db.get_atoms(id),vasp5=True)
#     data = fd.getvalue()
#     return data, '{}.vasp'.format(id)
#
#
#
#
# if download_button:
#     @app.route('/database/<project>/json')
#     @download
#     def jsonall(project):
#         con_id = int(request.args['x'])
#         # import pdb; pdb.set_trace()
#         con = connections[con_id]
#         data = tofile(project, con.query[2], 'json', con.limit)
#         return data, 'selection.json'
#
#
# @app.route('/database/<project>/json/<int:id>')
# @download
# def json1(project, id):
#     if project not in databases:
#         return 'No such project: ' + project, None
#     # import pdb; pdb.set_trace()
#     data = tofile(project, id, 'json')
#     return data, '{}.json'.format(id)
#
#
# if download_button:
#     @app.route('/database/<project>/sqlite')
#     @download
#     def sqliteall(project):
#         con_id = int(request.args['x'])
#         con = connections[con_id]
#         data = tofile(project, con.query[2], 'db', con.limit)
#         return data, 'selection.db'
#
#
# @app.route('/database/<project>/sqlite/<int:id>')
# @download
# def sqlite1(project, id):
#     if project not in databases:
#         return 'No such project: ' + project, None
#     data = tofile(project, id, 'db')
#     return data, '{}.db'.format(id)
#
#
#
#
# def pages(page, nrows, limit):
#     """Helper function for pagination stuff."""
#     npages = (nrows + limit - 1) // limit
#     p1 = min(5, npages)
#     p2 = max(page - 4, p1)
#     p3 = min(page + 5, npages)
#     p4 = max(npages - 4, p3)
#     pgs = list(range(p1))
#     if p1 < p2:
#         pgs.append(-1)
#     pgs += list(range(p2, p3))
#     if p3 < p4:
#         pgs.append(-1)
#     pgs += list(range(p4, npages))
#     pages = [(page - 1, 'previous')]
#     for p in pgs:
#         if p == -1:
#             pages.append((-1, '...'))
#         elif p == page:
#             pages.append((-1, str(p + 1)))
#         else:
#             pages.append((p, str(p + 1)))
#     nxt = min(page + 1, npages - 1)
#     if nxt == page:
#         nxt = -1
#     pages.append((nxt, 'next'))
#     return pages
#
# @app.route('/test_hyperlink')
# def test_hyperlink():
#     return render_template('hyperlink_fig.html')


if __name__ == '__main__':
    downloads = os.path.join(root, "downloads")
    if not os.path.isdir(downloads):
        os.makedirs(downloads)
    #connect_databases(['boron_sheet.db','In2O3.db'])
    #initialize_databases()
    app.run(host="0.0.0.0",port=2456,debug=True)

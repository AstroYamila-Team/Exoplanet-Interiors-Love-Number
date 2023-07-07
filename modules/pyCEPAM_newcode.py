import os
import re
import sys
import shutil
import string
import numpy as np
import subprocess32
from collections import OrderedDict

from mpi4py import MPI

# from pyMPI_util import *

class CodeWrapper(object):
    def __init__(self,adat=False,tree=False,mpi=False):
        if tree != False:
            if not(isinstance(tree,tuple)) :
                sys.exit(
                    """
                    Tree keyword must be a tuple
                    """
                )
            if len(tree) == 2:
                self._tree = self.set_tree(tree[0],files=tree[1])
            elif len(tree) == 1:
                self._tree = self.set_tree(tree[0])
            else:
                sys.exit(
                    """
                    Tree keyword must be a tuple of size 1 or 2
                    """
                )
                
        self.adat = adat
        self.mpi = mpi
        # print('The code is here')
        if self.mpi:
            self.make_tree = self._make_tree_mpi
            self.remove_tree = self._remove_tree_mpi
        else:
            # print('make tree definition happens')
            self.make_tree = self._make_tree_nompi
            self.remove_tree = self._remove_tree_nompi

    def set_tree(self,subdirs,files=[]):
        tree = (os.path.abspath('.'),subdirs,files)
        # print(tree)
        return tree

    def _make_tree_nompi(self):
        for dirs in self._tree[1]:
            if not(os.path.isdir(os.path.join(self._tree[0],dirs))):
                os.mkdir(os.path.join(self._tree[0],dirs))
            for ff in self._tree[2]:
                if os.path.isdir(ff):
                    if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
                        shutil.copytree(ff,os.path.join(self._tree[0],dirs,ff), symlinks=True)
                else:
                    if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
                        shutil.copy(ff,os.path.join(self._tree[0],dirs,ff), follow_symlinks=False)
            # for ff in ['./cepam.don','./cepam.don.conf', './cepam_atm.don', './cepam_atm.don.conf', './cepam_etat.don', './cepam_etat.don.conf', './cepam_figures.ini', './ctes_jup-j2mod.don', './ctes_jup-j2mod.don.conf', './mev.asc', './mev.basc', './mev.bin', './mev.bin.conf', './mev.conf', './jup.conf', './mev.ini', './mev.ini.conf', './evol.don', './chains', './cepam_flux.don', './cepam_lcentre.don', './cepam_lim_p14.don', './cepam_opa_ff_grains.don']:
            #         if os.path.isdir(ff):
            #             if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
            #                 shutil.copytree(ff,os.path.join(self._tree[0],dirs,ff), symlinks=True)
            #         else:
            #             if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
            #                 shutil.copy(ff,os.path.join(self._tree[0],dirs,ff), follow_symlinks=False)

    def _make_tree_mpi(self):
        commtree = MPI.COMM_WORLD
        ranktree = commtree.rank
        if ranktree == 0:
            for dirs in self._tree[1]:
                if not(os.path.isdir(os.path.join(self._tree[0],dirs))):
                    os.mkdir(os.path.join(self._tree[0],dirs))
                for ff in self._tree[2]:
                    if os.path.isdir(ff):
                        if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
                            shutil.copytree(ff,os.path.join(self._tree[0],dirs,ff), symlinks=True)
                    else:
                        if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
                            shutil.copy(ff,os.path.join(self._tree[0],dirs,ff), follow_symlinks=False)
                for ff in ['./cepam.don','./cepam.don.conf', './cepam_atm.don', './cepam_atm.don.conf', './cepam_etat.don', './cepam_etat.don.conf', './cepam_figures.ini', './ctes_jup-j2mod.don', './ctes_jup-j2mod.don.conf', './mev.asc', './mev.basc', './mev.bin', './mev.bin.conf', './mev.conf', './jup.conf', './mev.ini', './mev.ini.conf', './evol.don', './chains', './cepam_flux.don', './cepam_lcentre.don', './cepam_lim_p14.don', './cepam_opa_ff_grains.don']:
                    if os.path.isdir(ff):
                        if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
                            shutil.copytree(ff,os.path.join(self._tree[0],dirs,ff), symlinks=True)
                    else:
                        if not(os.path.exists(os.path.join(self._tree[0],dirs,ff))):
                            shutil.copy(ff,os.path.join(self._tree[0],dirs,ff), follow_symlinks=False)
        commtree.barrier()
        
    def clean_tree(self):
        pass

    def _remove_tree_mpi(self):
        commtree = MPI.COMM_WORLD
        ranktree = commtree.rank
        if ranktree == 0:
            for dirs in self._tree[1]:
                if os.path.isdir(os.path.join(self._tree[0],dirs)):
                    shutil.rmtree(os.path.join(self._tree[0],dirs))
        commtree.barrier()
        
    def _remove_tree_nompi(self):
        for dirs in self._tree[1]:
            if os.path.isdir(os.path.join(self._tree[0],dirs)):
                shutil.rmtree(os.path.join(self._tree[0],dirs))
        
    def model(self,p,o):
        pass

    def model_mult_MPI(self,p,o):
        pass

    def run(self):
        pass

    def clean(self):
        pass	

    def update(self,p):
        pass

    def get_current_vpar(self):
        pass

    def set_path(self,path):
        pass

    def get_obs(self,keys):
        pass
    
class PyCEPAM(CodeWrapper):
    _cepam_files = ['cepam.don','ctes_jup-j2mod.don','cepam_atm.don','cepam_etat.don','data','cepam_figures.ini', 'evol.don', 'cepam_lim_p14.don'] #List of the files used in this run
    _code_convert = { #Initial values?
        'mnoyau' :1.e-2,
        'yproto' :1.e-1,
        'yatm'   :1.e-1,
        'zatm'   :[1.e-1,1.e-1],
        'dz'	 :[1.e-1,1.e-1],
        'p_ppt'  :1.,
        'pd'	 :1.,
        'dro'	:1.e-2,
#		'tjump'  :1.e2,
        'T0'	 :1.e2,
        'yz'     :[1.e-1,1.e-1,1.e-1]
        }
    _pstatefmt = OrderedDict( #Setting up a dictionary for a lot of parameters.
        [
            ('files_eos',np.array([],dtype=str)),
            ('p_ppt',float()),
            ('icondense',int()),
            ('xcond',np.array([],dtype=float)),
            ('pd',float()),
            ('dp',float()),
            ('dro',float()),
            ('tjump',float()),
            ('dilcore',int()),
            ('p_dilcore',float()),
            ('dlogp_dilcore',float()),
            ('which_spec',int()),
            ('dspec_dilcore',float()),
            ('files_eos_aux',''),
            ('p_lp',float())
        ]
        )
    _cepam_state_struct = OrderedDict( #structure of the files, I think
        [
            ('NL_FILES',['files_eos']),
            ('NL_PPT',['p_ppt']),
            ('NL_CONDENSE',['icondense','xcond']),
            ('NL_DENSITE',['pd','dp','dro']),
            ('NL_TEMP_JUMP',['tjump']),
            ('NL_DILCORE',['dilcore']),
            ('NL_P_DILCORE',['p_dilcore','dlogp_dilcore']),
            ('NL_WHICH_SPEC',['which_spec']),
            ('NL_INCRE',['dspec_dilcore']),
            ('NL_FILES_AUX',['files_eos_aux']),
            ('NL_P_LP',['p_lp'])
            ]
        )
    _cepam_atm_struct = OrderedDict( # structure of the atm file
        [
            ('NL_ATM',['P0','T0'])
            ]
        )
    _patmfmt = OrderedDict( #One more
        [
            ('P0',float()),
            ('T0',float())
        ]
        )

    def __init__(self,
                 vpar,
                 pfx,
                 cepam_path='.',
                 subdirs=False,
                 mpi=False,
                 convert=False
    ):

        
        # Test (temporary)
        if np.array([x in vpar for x in ["files_eos",'files_eos_aux']]).any():
            sys.exit(
                """
                Variations of files_eos and files_eos_aux not yet implemented.
                Please remove it from the vpar list.
                """
            )
        # CEPAM files prefix
        self.pfx = pfx
        # CEPAM path setting
        self.cepam_path = cepam_path
        self.set_path(cepam_path)
        # Base value for the parameters taken from template files
        self.set_ini()
        self.set_cepam_state()
        self.set_cepam_atm()
        self.parr = [self._pini,self._pstate,self._patm]
        # Varying parameters
        ## Parameter rescaling
        self.convert = convert
        ## Parameter names
        self.vpar = vpar
        self._kobs = dict({
            'req':15, # was 17?
            'j2':20,
            'j4':23,
            'j6':26,
            'j8':29,
            'j10':32,
            'j12':35,
            })
        ## Lengths for the parameters: 1 for scalars, >1 for arrays
        self.set_vpar_sizes()
        # Set file names
        self._cepam_files = ['cepam.don','ctes_jup-j2mod.don','cepam_atm.don','cepam_etat.don','data','cepam_figures.ini', 'evol.don', 'cepam_lim_p14.don', 'jup.conf']
        if subdirs != False :
            self._set_cepam_files(cepam_bin=True)
            super(PyCEPAM,self).__init__(tree = (subdirs,self._cepam_files),mpi=mpi)
        else:
            self._set_cepam_files(cepam_bin=False)
            
    def _set_cepam_files(self,cepam_bin=False):
        # Puts the filenames in a list
        cepfsfx = ['.conf','.ini.conf','.bin.conf', '.bin', '.ini']#, '.don']
        # extra_ff = ['evol.don', 'cepam_lim_p14.don']
        for cc in cepfsfx: self._cepam_files.append(self.pfx+cc) 
        # for ff in extra_ff: self._cepam_files.append(ff)
        if cepam_bin:
            shutil.copy(os.path.join(self.cepam_path,'cepam'),'cepam')
            self._cepam_files.append('cepam')
            
    def model(self,p,o,full_output = False):
        #print(self._pini)
        self.update(p) #updates self._pini values to whatever's in p.
        #print(self._pini)
        self.set_bin()
        self.write_ini() #writes the files again, with self._pini values
        self.write_cepam_state()
        self.write_cepam_atm()
        #print('running cepam')
        self.update_mass_Teq_lum_P(self.mass_value, self.Teq_value, self.lum_value, self.P_value)
        ierr = self.run() #This goes to where cepam is executed.
        if ierr == 0:
            if full_output:
                omod,flt = self.get_obs(o,full_output = full_output)
            else:
                omod = self.get_obs(o)
        else:
            omod = np.empty(len(o))
            if full_output: flt = np.empty(len(self._kobs))
        #self.clean()
        if full_output:
            return omod, ierr, flt
        else:
            return omod, ierr


    def evolution_model(self, p, age_low, max_rows=None, minimum=False, full_output=False):
        self.update(p)
        self.set_bin()
        self.write_ini()
        self.update_mass_Teq(self.mass_value, self.Teq_value) 
        ierr = self.run(evol=True)
        if ierr == 0:
            omod = self.get_obs_lum(age_low, max_rows, minimum, full_output) 
        else:
            omod = np.empty(1)
        return omod, ierr

    def model_mult_MPI(self,p,o,full_output = False): #model multiple times to mpi
        comm = MPI.COMM_WORLD
        ncpu = comm.Get_size()
        rank = comm.Get_rank()
        name = MPI.Get_processor_name()
        #print('mult_mpi')
        # First implementation of the MPI multirun: number of processes need
        # to be equal to the number of subdirectories
        if len(self._tree[1]) < ncpu:
            sys.exit(
                """
                First implementation ncpu needs to be equals to len(subdirs)
                """
            )

        npar = len(p)
        nobs = len(o)
        ndat = len(self._kobs)
        #iproc = np.arange(npar)[rank::ncpu]
        #opart = np.empty(npar,dtype='O')[rank::ncpu]
        if full_output: dpart = np.empty(npar,dtype='O')[rank::ncpu]
        #ppart = p[rank::ncpu]
        #epart = np.empty(npar)[rank::ncpu]
        os.chdir(os.path.join(self._tree[0],self._tree[1][rank]))#Change to the dir to run it in. does this update the files in the correct locations?	
        #print(os.getcwd())
        if not os.path.exists("chains"): os.mkdir("chains")
        #print(os.path.join(self._tree[0],self._tree[1][rank]))
        #print(self._tree[1], self._tree[2], rank)
        #for j,pp in enumerate(ppart):
        if full_output:
            omod, ierr, dmod = self.model(p,o,full_output = full_output)
        else:
            omod, ierr = self.model(p,o)
        '''
        omod = MPI_combine(ncpu,rank,comm,opart,npar,dtype=object)
        omod = comm.bcast(omod,0)
        ierr = MPI_combine(ncpu,rank,comm,epart,npar)
        if full_output:
            dmod = MPI_combine(ncpu,rank,comm,dpart,npar,dtype=object)
            dmod = comm.bcast(dmod,0)
        ierr = comm.bcast(ierr,0)
        '''
        if full_output:
            return omod, ierr, dmod
        else:
            return omod, ierr


### !! This is where cepam is
    def run(self, evol=False):
        #print('actually running')
        # shutil.copy(self.pfx+'.bin.conf',self.pfx+'.bin')
        try:
            ierr = os.system("timeout 45s ./cepam < jup.conf 1> log1 2> log2")
            if evol:
                ierr2 = os.system('timeout 45s ./cepam < mev.conf 1> log1 2> log2')
                if ierr2 != 0:
                    print("In running s/r (evolution) CEPAM call error return", ierr2)
                    # ierr = ierr2
                else:
                    print('ierr(evolution):', ierr2)
                    ierr = ierr2
            #ierr = os.system("timeout 45s ./cepam < jup.conf 1> log1 2> log2")
            #ierr = os.system("./cepam < mev.conf 1> log1 2> log2")
            if ierr != 0:
                tmpbin = bin(ierr)[2:]
                ierr = int(tmpbin[:8],2)
                skil = int(tmpbin[8:],2)
                print('In running s/r CEPAM call error return',ierr)
            else:
                print('ierr: ', ierr)
            return ierr
        except:
            print("Problem in CEPAM run, setting error to 1")
            return 1
        

    def clean(self):
        clean_list = [
            self.pfx+sfx for sfx in [
                '.ini','_js.csv','.asc','.des','_n.bin','.csv','.bin'
            ]
        ]
        clean_list = clean_list + ['cepam_etat.don']
        clean_list = clean_list + ['cepam_atm.don']
        for c in clean_list:
            try:
                os.remove(c)
            except:
                pass
                #print "File %s not found"%(c)		

    def set_bin(self):
        shutil.copyfile('mev.bin.conf','mev.bin')
        
    def set_vpar_sizes(self):
        self.vpar_sz = []
        # print([x for x in self.parr])
        for vv in self.vpar:
                # print(vv)
                # print(np.where([vv in x for x in self.parr]))
                ip = np.where([vv in x for x in self.parr])[0][0]
                if np.logical_or(
                        type(self.parr[ip][vv]) == list,
                        type(self.parr[ip][vv]) == np.ndarray
                ):
                    self.vpar_sz.append(len(self.parr[ip][vv]))
                else:
                    self.vpar_sz.append(1)
        self.vpar_len = sum(self.vpar_sz)

    def update(self,p):
        # Test parameter vector length
        if len(p) != self.vpar_len:
            sys.exit(
                """
                Parameter vector length does not match vpar length.
                """
            )
        parr = [self._pini,self._pstate]
        i = 0
        for vv,j in zip(self.vpar,self.vpar_sz):
                ip = np.where([vv in x for x in self.parr])[0][0]
                if j == 1:
                    if self.convert:
                        self.parr[ip][vv] = p[i]*self._code_convert[vv]
                    else:
                        self.parr[ip][vv] = p[i]
                    i+=1
                else:
                    for jj in range(j):
                        if self.convert:
                            self.parr[ip][vv][jj] = p[i]*self._code_convert[vv][jj]
                        else:
                            self.parr[ip][vv][jj] = p[i]
                        i+=1

    def get_current_vpar(self):
        crtvpar = []
        parr = [self._pini,self._pstate]
        i = 0
        for vv,j in zip(self.vpar,self.vpar_sz):
                ip = np.where([vv in x for x in self.parr])[0][0]
                if j == 1:
                    crtvpar.append(self.parr[ip][vv])
                    i+=1
                else:
                    for jj in range(j):
                        crtvpar.append(self.parr[ip][vv][jj])
                        i+=1
        return np.array(crtvpar)
    
    def set_path(self,path):
        if os.environ.get('CEPAM_PYTHON') == None:
            os.environ['CEPAM_PYTHON'] = path
        regexp = re.compile(path)
        if not(regexp.search(os.environ['PATH'])):
            os.environ['PATH'] = ":".join(
                [os.environ['PATH'],os.environ['CEPAM_PYTHON']]
            )
    def get_obs_old(self,keys,full_output = False):
        djs = np.loadtxt(self.pfx+'_js.csv',delimiter=',')
        #print(self.pfx+'_js.csv')
        if full_output:
            return {kk:djs[self._kobs[kk]] for kk in keys},{kk:djs[self._kobs[kk]] for kk in self._kobs.keys()}
        else:
            return {kk:djs[self._kobs[kk]] for kk in keys}
            
    def get_obs(self,keys,full_output = False):
        djs = np.loadtxt('mev_js.csv',delimiter=',')
        #print(os.getcwd())
        #print(djs)
        #print(self.pfx+'_js.csv')
        if full_output:
            return {kk:djs[self._kobs[kk]] for kk in keys},{kk:djs[self._kobs[kk]] for kk in self._kobs.keys()}
        else:
            return {kk:djs[self._kobs[kk]] for kk in keys}#{'req':djs[15]}

    def get_obs_lum(self, age_lim, max_rows=None, minimum=False, full_output=False):
        output = np.loadtxt('mev_ev.csv', delimiter=',', skiprows=1, max_rows=max_rows)
        # print(output)
        age = output[:, 0]
        lum = output[:, 2]
        mask=(age>=age_lim)
        if minimum:
            mask = (age<=age_lim)
            try: 
                lum_min = np.min(lum[mask])
            except:
                lum_min = np.min(lum)
                print('lum does not correspond to age')
            return {'lum_min': lum_min}
        elif full_output:
            return (age, lum)
        else:
            try:
                lum_max = np.max(lum[mask])
            except:
                lum_max = np.min(lum) # why min?
                print('lum does not correspond to age')
            return {'lum_max': lum_max}

    def _set_fmt_ini(self):
        f = open(self.pfx+'.ini.conf','r') # Opening jup.ini.conf and reading in the parameters in there
        for ff in f:
            if not(ff[0] == '#'): #If the line if not commented out
                if re.match('comp_type',ff): #If the line has comp_type in it
                    case = re.search('["\'](.+)["\']$',ff).groups()[0] #Search for a string
                    if case == "Y&dZ discontinuous":						
                        self._pinifmt = OrderedDict(
                            [
                                ('mplanet',float()),
                                ('mnoyau',float()),
                                ('p_ice',float()),
                                ('omega',float()),
                                ('comp_type',''),
                                ('nbelem',int()),
                                ('yproto',float()),
                                ('yatm',float()),
                                ('zatm',[]),
                                ('dz',[])
                            ]
                        )
                    elif case == "Y&Z discontinuous":						
                        self._pinifmt = OrderedDict(
                            [
                                ('mplanet',float()),
                                ('mnoyau',float()),
                                ('p_ice',float()),
                                ('omega',float()),
                                ('comp_type',''),
                                ('nbelem',int()),
                                ('yproto',float()),
                                ('yatm',float()),
                                ('zatm',[]),
                                ('zdeep',[])
                            ]
                        )
                    elif case == "homogeneous":						
                        self._pinifmt = OrderedDict(
                            [
                                ('mplanet',float()),
                                ('mnoyau',float()),
                                ('p_ice',float()),
                                ('omega',float()),
                                ('comp_type',''),
                                ('nbelem',int()),
                                ('yz',[]),

                            ]
                        )
                if re.match('core_type',ff):
                    case = re.search('["\'](.+)["\']$',ff).groups()[0]
                    if case == "dilute":
                        self._pinifmt.update(
                            {
                                'core_type' : '',
                                'mdilute' : float(),
                                'deltamdil' : float(),
                                'zdilute' : [],
                                'ydilute' : int(),
                                'fadiabatic' : float()
                                }
                            )
                    elif case == "":
                        self._pinifmt.update(
                            {
                                'core_type' : ''
                                }
                            )
                        
        f.close()

    def update_mass_Teq(self, mass, Teq):
        #print('update mass')
        jup_mass = 1.89831e30 #g
        temp=Teq
        mass_str_lims = [1.e25, 1.e26, 1.e27, 1.e28, 1.e29, 1.e30, 1.e31, 1.e32, 1.e33, 1.e34]
        mass_str_strings = ['D+25','D+26','D+27','D+28', 'D+29', 'D+30', 'D+31', 'D+32', 'D+33', 'D+34']
        mass_step = mass-np.array(mass_str_lims)
        mass_step_2 = mass_step[np.argwhere(mass_step>0)]
        mass_closest = np.min(mass_step_2)
        arg_min_mass = np.argmin(np.abs(mass_closest-np.array(mass_step)))
        mass_arg= arg_min_mass
        mass_closest_true = mass_str_lims[mass_arg]

        mass_str = '{0:.8f}'.format(np.float64(mass)/mass_closest_true) + mass_str_strings[mass_arg]
        
        temp_str_lims = [1.e0, 1.e1, 1.e2, 1.e3, 1.e4]
        temp_str_strings = ['D+00','D+01','D+02','D+03', 'D+04']
        temp_step = temp-np.array(temp_str_lims)
        temp_step_2 = temp_step[np.argwhere(temp_step>0)]
        temp_closest = np.min(temp_step_2)
        arg_min_temp = np.argmin(np.abs(temp_closest-np.array(temp_step)))
        temp_arg= arg_min_temp
        temp_closest_true = temp_str_lims[temp_arg]

        temp_str = '{0:.8f}'.format(np.float64(temp)/temp_closest_true) + temp_str_strings[temp_arg]

        file = open('./ctes_jup-j2mod.don', "r")
        new_file=''
        for line in file:
            if line.startswith('Masse:              '):
                new_line = 'Masse:              '+mass_str+' g             [1]   '+os.linesep
                #print(new_line)
            elif line.startswith('Temp. equilibre:   '):
                new_line = 'Temp. equilibre:    '+temp_str+' K             [2]'+os.linesep
                #print(new_line)
            else:
                new_line=line
            new_file+=new_line
        file.close()
        # opening the file in write mode
        fout = open('./ctes_jup-j2mod.don', "w")
        fout.write(new_file)
        fout.close()

    def update_mass_Teq_lum_P(self, mass, Teq, lum, P):
        #print('update mass')
        jup_mass = 1.89831e30 # g
        s_in_day = 24 * 3600 # s
        temp=Teq
        P_in_s = P * s_in_day

        mass_str_lims = [1.e25, 1.e26, 1.e27, 1.e28, 1.e29, 1.e30, 1.e31, 1.e32, 1.e33, 1.e34]
        mass_str_strings = ['D+25','D+26','D+27','D+28', 'D+29', 'D+30', 'D+31', 'D+32', 'D+33', 'D+34']
        mass_step = mass-np.array(mass_str_lims)
        mass_step_2 = mass_step[np.argwhere(mass_step>0)]
        mass_closest = np.min(mass_step_2)
        arg_min_mass = np.argmin(np.abs(mass_closest-np.array(mass_step)))
        mass_arg= arg_min_mass
        mass_closest_true = mass_str_lims[mass_arg]

        mass_str = '{0:.8f}'.format(np.float64(mass)/mass_closest_true) + mass_str_strings[mass_arg]
        
        temp_str_lims = [1.e0, 1.e1, 1.e2, 1.e3, 1.e4]
        temp_str_strings = ['D+00','D+01','D+02','D+03', 'D+04']
        temp_step = temp-np.array(temp_str_lims)
        temp_step_2 = temp_step[np.argwhere(temp_step>0)]
        temp_closest = np.min(temp_step_2)
        arg_min_temp = np.argmin(np.abs(temp_closest-np.array(temp_step)))
        temp_arg= arg_min_temp
        temp_closest_true = temp_str_lims[temp_arg]

        temp_str = '{0:.8f}'.format(np.float64(temp)/temp_closest_true) + temp_str_strings[temp_arg]

        P_str_lims = [1.e3, 1.e4, 1.e5, 1.e6, 1.e7]
        P_str_strings = ['D+03', 'D+04', 'D+05', 'D+06', 'D+07']
        P_step = P_in_s - np.array(P_str_lims)
        P_step_2 = P_step[np.argwhere(P_step>0)]
        P_closest = np.min(P_step_2)
        arg_min_P = np.argmin(np.abs(P_closest-np.array(P_step)))
        P_arg = arg_min_P
        P_closest_true = P_str_lims[P_arg]
        
        P_str = '{0:.8f}'.format(np.float64(P_in_s)/P_closest_true) + P_str_strings[P_arg]

        lum_str_lims = [1.e18, 1.e19, 1.e20, 1.e21, 1.e22, 1.e23, 1.e24, 1.e25, 1.e26, 1.e27, 1.e28, 1.e29, 1.e30, 1.e31, 1.e32]
        lum_str_strings = ['D+18','D+19','D+20','D+21', 'D+22', 'D+23', 'D+24', 'D+25', 'D+26', 'D+27', 'D+28', 'D+29', 'D+30', 'D+31', 'D+32']
        lum_step = lum-np.array(lum_str_lims)
        lum_step_2 = lum_step[np.argwhere(lum_step>0)]
        lum_closest = np.min(lum_step_2)
        arg_min_lum = np.argmin(np.abs(lum_closest-np.array(lum_step)))
        lum_arg= arg_min_lum
        lum_closest_true = lum_str_lims[lum_arg]

        lum_str = '{0:.8f}'.format(np.float64(lum)/lum_closest_true) + lum_str_strings[lum_arg]

        file = open('./ctes_jup-j2mod.don', "r")
        new_file=''
        for line in file:
            if line.startswith('Masse:              '):
                new_line = 'Masse:              '+mass_str+' g             [1]   '+os.linesep
                #print(new_line)
            elif line.startswith('Luminosite interne: '):
                new_line = 'Luminosite interne: '+lum_str+ ' erg/s         [2]'+os.linesep
                #print(new_line)
            elif line.startswith('Temp. equilibre:   '):
                new_line = 'Temp. equilibre:    '+temp_str+' K             [2]'+os.linesep
                #print(new_line)
            elif line.startswith('Periode rotation:   '):
                new_line = 'Periode rotation:   '+P_str+' sec           [30days]'+os.linesep
            else:
                new_line=line
            new_file+=new_line
        file.close()
        # opening the file in write mode
        fout = open('./ctes_jup-j2mod.don', "w")
        fout.write(new_file)
        fout.close()

    def update_mass_Teq_lum(self, mass, Teq, lum):
        #print('update mass')
        jup_mass = 1.89831e30 #g
        temp=Teq
        mass_str_lims = [1.e25, 1.e26, 1.e27, 1.e28, 1.e29, 1.e30, 1.e31, 1.e32, 1.e33, 1.e34]
        mass_str_strings = ['D+25','D+26','D+27','D+28', 'D+29', 'D+30', 'D+31', 'D+32', 'D+33', 'D+34']
        mass_step = mass-np.array(mass_str_lims)
        mass_step_2 = mass_step[np.argwhere(mass_step>0)]
        mass_closest = np.min(mass_step_2)
        arg_min_mass = np.argmin(np.abs(mass_closest-np.array(mass_step)))
        mass_arg= arg_min_mass
        mass_closest_true = mass_str_lims[mass_arg]

        mass_str = '{0:.8f}'.format(np.float64(mass)/mass_closest_true) + mass_str_strings[mass_arg]
        
        temp_str_lims = [1.e0, 1.e1, 1.e2, 1.e3, 1.e4]
        temp_str_strings = ['D+00','D+01','D+02','D+03', 'D+04']
        temp_step = temp-np.array(temp_str_lims)
        temp_step_2 = temp_step[np.argwhere(temp_step>0)]
        temp_closest = np.min(temp_step_2)
        arg_min_temp = np.argmin(np.abs(temp_closest-np.array(temp_step)))
        temp_arg= arg_min_temp
        temp_closest_true = temp_str_lims[temp_arg]

        temp_str = '{0:.8f}'.format(np.float64(temp)/temp_closest_true) + temp_str_strings[temp_arg]
        
        lum_str_lims = [1.e18, 1.e19, 1.e20, 1.e21, 1.e22, 1.e23, 1.e24, 1.e25, 1.e26, 1.e27, 1.e28, 1.e29, 1.e30, 1.e31, 1.e32]
        lum_str_strings = ['D+18','D+19','D+20','D+21', 'D+22', 'D+23', 'D+24', 'D+25', 'D+26', 'D+27', 'D+28', 'D+29', 'D+30', 'D+31', 'D+32']
        lum_step = lum-np.array(lum_str_lims)
        lum_step_2 = lum_step[np.argwhere(lum_step>0)]
        lum_closest = np.min(lum_step_2)
        arg_min_lum = np.argmin(np.abs(lum_closest-np.array(lum_step)))
        lum_arg= arg_min_lum
        lum_closest_true = lum_str_lims[lum_arg]

        lum_str = '{0:.8f}'.format(np.float64(lum)/lum_closest_true) + lum_str_strings[lum_arg]
        file = open('./ctes_jup-j2mod.don.conf', "r")
        new_file=''
        for line in file:
            if line.startswith('Masse:              '):
                new_line = 'Masse:              '+mass_str+' g             [1]   '+os.linesep
                #print(new_line)
            elif line.startswith('Luminosite interne: '):
                new_line = 'Luminosite interne: '+lum_str+ ' erg/s         [2]'+os.linesep
                #print(new_line)
            elif line.startswith('Temp. equilibre:   '):
                new_line = 'Temp. equilibre:    '+temp_str+' K             [2]'+os.linesep
                #print(new_line)
            else:
                new_line=line
            new_file+=new_line
        file.close()
        # opening the file in write mode
        fout = open('./ctes_jup-j2mod.don', "w")
        fout.write(new_file)
        fout.close()

    # def update_mass_Teq_lum_J2(self, mass, Teq, lum, J2):
    #     #print('update mass')
    #     jup_mass = 1.89831e30 #g
    #     temp=Teq
    #     mass_str_lims = [1.e25, 1.e26, 1.e27, 1.e28, 1.e29, 1.e30, 1.e31, 1.e32, 1.e33, 1.e34]
    #     mass_str_strings = ['D+25','D+26','D+27','D+28', 'D+29', 'D+30', 'D+31', 'D+32', 'D+33', 'D+34']
    #     mass_step = mass-np.array(mass_str_lims)
    #     mass_step_2 = mass_step[np.argwhere(mass_step>0)]
    #     mass_closest = np.min(mass_step_2)
    #     arg_min_mass = np.argmin(np.abs(mass_closest-np.array(mass_step)))
    #     mass_arg= arg_min_mass
    #     mass_closest_true = mass_str_lims[mass_arg]

    #     mass_str = '{0:.8f}'.format(np.float64(mass)/mass_closest_true) + mass_str_strings[mass_arg]
        
    #     temp_str_lims = [1.e0, 1.e1, 1.e2, 1.e3, 1.e4]
    #     temp_str_strings = ['D+00','D+01','D+02','D+03', 'D+04']
    #     temp_step = temp-np.array(temp_str_lims)
    #     temp_step_2 = temp_step[np.argwhere(temp_step>0)]
    #     temp_closest = np.min(temp_step_2)
    #     arg_min_temp = np.argmin(np.abs(temp_closest-np.array(temp_step)))
    #     temp_arg= arg_min_temp
    #     temp_closest_true = temp_str_lims[temp_arg]

    #     temp_str = '{0:.8f}'.format(np.float64(temp)/temp_closest_true) + temp_str_strings[temp_arg]
        
    #     lum_str_lims = [1.e18, 1.e19, 1.e20, 1.e21, 1.e22, 1.e23, 1.e24, 1.e25, 1.e26, 1.e27, 1.e28, 1.e29, 1.e30, 1.e31, 1.e32]
    #     lum_str_strings = ['D+18','D+19','D+20','D+21', 'D+22', 'D+23', 'D+24', 'D+25', 'D+26', 'D+27', 'D+28', 'D+29', 'D+30', 'D+31', 'D+32']
    #     lum_step = lum-np.array(lum_str_lims)
    #     lum_step_2 = lum_step[np.argwhere(lum_step>0)]
    #     lum_closest = np.min(lum_step_2)
    #     arg_min_lum = np.argmin(np.abs(lum_closest-np.array(lum_step)))
    #     lum_arg= arg_min_lum
    #     lum_closest_true = lum_str_lims[lum_arg]

    #     lum_str = '{0:.8f}'.format(np.float64(lum)/lum_closest_true) + lum_str_strings[lum_arg]

    #     J2_str_lims = [1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1]
    #     J2_str_strings = ['D-03', 'D-02', 'D-01', 'D+00', 'D+01']
    #     J2_step = J2 - np.array(J2_str_lims)
    #     J2_step2 = J2_step[np.argwhere(J2_step>0)]
    #     J2_closest = np.min(J2_step2)
    #     arg_min_J2 = np.argmin(np.abs(J2_closest - np.array(J2_step)))
    #     J2_closest_true = J2_str_strings[arg_min_J2]

    #     J2_str = '{0:.8f}'.format(np.float64(J2)/J2_closest_true) + J2_str_strings[arg_min_J2]

    #     file = open('./ctes_jup-j2mod.don.conf', "r")
    #     new_file=''
    #     for line in file:
    #         if line.startswith('Masse:              '):
    #             new_line = 'Masse:              '+mass_str+' g             [1]   '+os.linesep
    #             #print(new_line)
    #         elif line.startswith('Luminosite interne: '):
    #             new_line = 'Luminosite interne: '+lum_str+ ' erg/s         [2]'+os.linesep
    #             #print(new_line)
    #         elif line.startswith('Temp. equilibre:   '):
    #             new_line = 'Temp. equilibre:    '+temp_str+' K             [2]'+os.linesep
    #             #print(new_line)
    #         elif line.startswith('    J2:'):
    #             new_line = '    J2:             '+J2_str+'               [7]'+os.linesep
    #         else:
    #             new_line=line
    #         new_file+=new_line
    #     file.close()
    #     # opening the file in write mode
    #     fout = open('./ctes_jup-j2mod.don', "w")
    #     fout.write(new_file)
    #     fout.close()
    # # def set_ini2(self, p):

    def set_ini(self):
        self._set_fmt_ini()
        self._pini = self._pinifmt.copy()
        # print(self._pini)
        f = open(self.pfx+'.ini.conf','r')
        for i,ff in enumerate(f):
            if not(ff[0] == '#'):
                ktst = [re.match(pp,ff) for pp in self._pini.keys()]
                try:
                    ik = np.where(ktst)[0][0]
                except:
                    errkwd = re.match('(\w+)\s*',ff)
                    raise Exception("At line {0:d}: keyword '{1:s}' not defined.".format(i,errkwd.groups()[0]))
                k = ktst[ik].group()
                if type(self._pini[k]) == float:
                    self._pini[k] = float(ff.split()[2])
                elif type(self._pini[k]) == int:
                    #print('here', self._pini[k], k)
                    self._pini[k] = int(ff.split()[2])
                elif type(self._pini[k]) == list:
                    #print('here', self._pini[k], k)
                    self._pini[k] = [
                        float(x) for x in ff.split()[2].split(',')
                    ]
                elif type(self._pini[k]) == str:
                    self._pini[k] = re.search('["\'](.+)["\']$',ff).groups()[0]
        f.close()
            
    def write_ini(self):
        f = open(self.pfx+'.ini','w')
        for kk in self._pini.keys():
            # print(self._pini)
            if type(self._pinifmt[kk]) == float:
                f.write("%s = %10.8f\n"%(kk,self._pini[kk]))
            if type(self._pinifmt[kk]) == int:
                f.write("%s = %2d\n"%(kk,self._pini[kk]))
            elif type(self._pinifmt[kk]) == str:
                f.write("%s = %s\n"%(kk,'"'+self._pini[kk]+'"'))
            elif type(self._pinifmt[kk]) == list:
                fmt = "%s = "+','.join(['%10.8f' for xx in self._pini[kk]])+"\n"
                f.write(fmt%(tuple([kk]+[xx for xx in self._pini[kk]])))
        f.close()
                
    def set_cepam_state(self):
        self._pstate = self._pstatefmt.copy()
        f = open('cepam_etat.don.conf','r')
        for ff in f:
            if not(ff[0] == '&'):
                ktst = [re.match(pp,ff[1:]) for pp in self._pstate.keys()]
                kloc = np.where(ktst)[0]
                #print(self._pstate.keys(),ff[1:])
                kloc = np.where([re.match(pp,ff[1:]) for pp in self._pstate.keys()])[0]
                if len(kloc) == 1:
                    ik =  kloc[0]
                else:
                    ik = kloc[1]
                k = ktst[ik].group()
                    
                if type(self._pstate[k]) == float:
                    self._pstate[k] = float(ff.split()[2])
                elif type(self._pstate[k]) == int:
                    self._pstate[k] = int(ff.split()[2])
                elif type(self._pstate[k]) == str:
                    self._pstate[k] = re.search('["\'](.+)["\']/$',ff).groups()[0]
                elif type(self._pstate[k]) == np.ndarray:
                    if self._pstate[k].dtype == 'U1':
                        # A FAIRE: re-arranger les expressions regulieres pour permettre des espaces avant les / en fin de ligne
                        self._pstate[k] = [re.search('["\'](.+)["\']$',x).groups()[0] for x in ff.split()[2][:-1].split(',')]
                    elif self._pstate[k].dtype == 'float':
                        self._pstate[k] = [
                            float(x) for x in ff.split()[2].split(',')
                                           ]						
        f.close()

    def write_cepam_state(self):
        f = open('cepam_etat.don','w')
        for kk in self._cepam_state_struct.keys():
            f.write('&%s\n'%(kk))
            lk = len(self._cepam_state_struct[kk])
            end_line_mark = [',' for ik in range(lk - 1)]+['/']
            for kkk,em in zip(self._cepam_state_struct[kk],end_line_mark):
                if type(self._pstatefmt[kkk]) == float:
                    f.write(" %s = %10.8f %s\n"%(
                        kkk,self._pstate[kkk],em))
                elif type(self._pstatefmt[kkk]) == int: 
                    f.write(" %s = %2d %s\n"%(
                        kkk,self._pstate[kkk],em))
                elif type(self._pstatefmt[kkk]) == str: 
                    f.write(" %s = '%s' %s\n"%(
                        kkk,self._pstate[kkk],em))
                elif type(self._pstatefmt[kkk]) == np.ndarray:
                    if self._pstatefmt[kkk].dtype == 'U1':
                       fmt = "%s = "+','.join(["'%s'" for xx in self._pstate[kkk]])+" %s\n"%(em)
                    if self._pstatefmt[kkk].dtype == 'float':
                       fmt = "%s = "+','.join(['%10.8f' for xx in self._pstate[kkk]])+" %s\n"%(em)
                    f.write(fmt%(
                        tuple(
                            [kkk]+[xx for xx in self._pstate[kkk]]
                        )
                    ))
        f.close()

        
    def set_cepam_atm(self):
        self._patm = self._patmfmt.copy()
        f = open('cepam_atm.don.conf','r')
        for ff in f:
            if not(ff[0] == '&'):
                ktst = [re.match(pp,ff[1:]) for pp in self._patm.keys()]
                kloc = np.where(ktst)[0]
                if len(kloc) == 1:
                    ik =  kloc[0]
                else:
                    ik = kloc[1]					
                k = ktst[ik].group()
                if type(self._patm[k]) == float:
                    self._patm[k] = float(ff.split()[2])
                elif type(self._patm[k]) == int:
                    self._patm[k] = int(ff.split()[2])
                elif type(self._patm[k]) == str:
                    self._patm[k] = re.search('["\'](.+)["\']/$',ff).groups()[0]
                elif type(self._patm[k]) == np.ndarray:
                    if self._patm[k].dtype == 'S1':
                        self._patm[k] = [re.search('["\'](.+)["\']$',x).groups()[0] for x in string.split(string.split(ff)[2][:-1],',')]
                    elif self._patm[k].dtype == 'float':
                        self._patm[k] = [
                            float(x) for x in ff.split()[2].split(',')
                            ]

        f.close()

    def write_cepam_atm(self):
        f = open('cepam_atm.don','w')
        for kk in self._cepam_atm_struct.keys():
            f.write('&%s\n'%(kk))
            lk = len(self._cepam_atm_struct[kk])
            end_line_mark = [',' for ik in range(lk - 1)]+['/']
            for kkk,em in zip(self._cepam_atm_struct[kk],end_line_mark):
                if type(self._patmfmt[kkk]) == float:
                    f.write(" %s = %10.8f %s\n"%(
                        kkk,self._patm[kkk],em))
                elif type(self._patmfmt[kkk]) == int: 
                    f.write(" %s = %2d %s\n"%(
                        kkk,self._patm[kkk],em))
                elif type(self._patmfmt[kkk]) == str: 
                    f.write(" %s = '%s' %s\n"%(
                        kkk,self._patm[kkk],em))
                elif type(self._patmfmt[kkk]) == np.ndarray:
                    if self._patmfmt[kkk].dtype == 'S1':
                       fmt = "%s = "+','.join(["'%s'" for xx in self._patm[kkk]])+" %s\n"%(em)
                    if self._patmfmt[kkk].dtype == 'float':
                       fmt = "%s = "+','.join(['%10.8f' for xx in self._patm[kkk]])+" %s\n"%(em)
                    f.write(fmt%(
                        tuple(
                            [kkk]+[xx for xx in self._patm[kkk]]
                        )
                    ))
        f.close()




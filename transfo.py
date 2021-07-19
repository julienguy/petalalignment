import numpy as np
import scipy.optimize as opt
import json
import yaml

class Transfo3D() :
    '''
    Function to minimize when best fitting the transform matrix between input
    BMR locations and target BMR locations.

    mat: 1x6 Numpy array, with the first 3 values being a target rotation vector
    and the last 3 values a translation vector

    inputs: 3xN Numpy array containing measured BMR locations, where N is the
    number of Bmr

    targets: 3xN Numpy array containing target BMR locations
    '''

    # rotation then translation
    def __init__(self) :
        self.mirror    = False
        self.ax = 0.
        self.ay = 0.
        self.az = 0.
        self.t  = np.zeros(3,dtype=float)

    def rotmat(self) :
        cx=np.cos(self.ax)
        sx=np.sin(self.ax)
        cy=np.cos(self.ay)
        sy=np.sin(self.ay)
        cz=np.cos(self.az)
        sz=np.sin(self.az)

        mx = np.array([[1,0,0],[0,cx,sx],[0,-sx,cx]])
        my = np.array([[cy,0,-sy],[0,1,0],[sy,0,cy]])
        mz = np.array([[cz,sz,0],[-sz,cz,0],[0,0,1]])
        m = mx.dot(my).dot(mz)
        return m

    def apply(self,xyz) :

        xyz = np.atleast_1d(xyz).astype(float)
        assert(xyz.shape[0]==3)

        res = xyz.copy()
        if len(xyz.shape)==1 :
            res += self.t
        else :
            res += self.t[:,None]

        if self.mirror :
            res[0] *= -1
        res = self.rotmat().dot(res)
        return res

    def fit(self,input_xyz,target_xyz) :

        def setparams(params) :
            self.ax = params[0]
            self.ay = params[1]
            self.az = params[2]
            self.t  = params[3:6]

        def chi2(params, input_xyz, target_xyz) :
            setparams(params)
            transformed_xyz = self.apply(input_xyz)
            return np.sum ( (target_xyz-transformed_xyz)**2 )


        diff=np.mean(target_xyz-input_xyz,axis=1)
        self.mirror = False
        fits = opt.minimize(chi2, [0, 0, 1, diff[0], diff[1], diff[2]],
                               args = (input_xyz,target_xyz))
        params_direct = fits.x
        chi2_direct = chi2(params_direct,input_xyz,target_xyz)

        test_mirror = True

        if test_mirror :
            self.mirror = True
            fits = opt.minimize(chi2, [0, 0, 1, diff[0], diff[1], diff[2]],
                                args = (input_xyz,target_xyz))
            params_mirror = fits.x
            chi2_mirror = chi2(params_mirror,input_xyz,target_xyz)

            if chi2_mirror < chi2_direct :
                self.mirror = True
                setparams(params_mirror)
            else :
                self.mirror = False
                setparams(params_direct)
        else :
            setparams(params_direct)

        # compute rms distance
        transformed_xyz = self.apply(input_xyz)
        rms = np.sqrt(np.sum((target_xyz-transformed_xyz)**2)/target_xyz.shape[1])
        return rms

    def __str__(self) :
        r2d=180/np.pi
        line="Transfo3D: rotation angles x={:.1f} y={:.1f} z={:.1f} deg\n".format(r2d*self.ax,r2d*self.ay,r2d*self.az)
        line+="Transfo3D: translation dx={:.3f} dy={:.3f} dz={:.3f}\n".format(self.t[0],self.t[1],self.t[2])
        line+="Transfo3D: mirror = {}".format(self.mirror)
        return line

    @classmethod
    def read_dict(cls,params):
        tx = cls()
        tx.ax    = params['ax']
        tx.ay    = params['ay']
        tx.az    = params['az']
        tx.t[0]    = params['dx']
        tx.t[1]    = params['dy']
        tx.t[2]    = params['dz']
        tx.mirror= params['mirror']
        return tx

    def write_dict(self):
        params = dict()
        params['ax'] = float(self.ax)
        params['ay'] = float(self.ay)
        params['az'] = float(self.az)
        params['dx'] = float(self.t[0])
        params['dy'] = float(self.t[1])
        params['dz'] = float(self.t[2])
        params['mirror'] = bool(self.mirror)
        return params

    @classmethod
    def read_json(cls,filename):
        import json
        with open(filename) as fx:
            s = fx.read()
        tx = cls()
        params  = json.loads(s)
        return cls.read_dict(params)

    @classmethod
    def read_yaml(cls,filename):
        import yaml
        with open(filename) as fx:
            s = fx.read()
        params  = yaml.safe_load(s)
        return cls.read_dict(params)

    @classmethod
    def read(cls,filename):
        if filename.endswith("yaml"):
            return cls.read_yaml(filename)
        elif filename.endswith("json"):
            return cls.read_json(filename)
        else :
            raise IOError("Cannot write transfo to '{}'. Only yaml or json are supported.".format(filename))

    def write_json(self, filename):
        import json
        params = self.write_dict()
        with open(filename, 'w') as fx:
            fx.write(json.dumps(params))

    def write_yaml(self, filename):
        import yaml
        params = self.write_dict()
        with open(filename, 'w') as fx:
            fx.write(yaml.dump(params))

    def write(self, filename):
        if filename.endswith("yaml"):
            self.write_yaml(filename)
        elif filename.endswith("json"):
            self.write_json(filename)
        else :
            raise IOError("Cannot write transfo to '{}'. Only yaml or json are supported.".format(filename))

class Transfo2D() :
    '''
    Function to minimize when best fitting the transform matrix between input
    BMR locations and target BMR locations.

    mat: 1x6 Numpy array, with the first 3 values being a target rotation vector
    and the last 3 values a translation vector

    inputs: 3xN Numpy array containing measured BMR locations, where N is the
    number of Bmr

    targets: 3xN Numpy array containing target BMR locations
    '''

    # rotation then translation
    def __init__(self) :
        self.mirror    = False
        self.angle = 0
        self.t  = np.zeros(3)

    def rotmat(self) :
        ca=np.cos(self.angle)
        sa=np.sin(self.angle)
        return np.array([[ca,sa,0],[-sa,ca,0],[0,0,1]])

    def apply(self,xyz) :

        xyz = np.atleast_1d(xyz)
        assert(xyz.shape[0]==3)
        res = self.rotmat().dot(xyz)
        if self.mirror :
            res[0] *= -1
        if len(xyz.shape)==1 :
            res += self.t
        else :
            res += self.t[:,None]
        return res

    def fit(self,input_xyz,target_xyz) :

        def chi2(params, input_xyz, target_xyz) :
            self.angle = params[0]
            self.t[:2]  = params[1:3]
            transformed_xyz = self.apply(input_xyz)
            return np.sum ( (target_xyz-transformed_xyz)**2 )

        self.mirror = False
        fits = opt.minimize(chi2, [0, 0, 0],
                               args = (input_xyz,target_xyz))
        params_direct = fits.x
        chi2_direct = chi2(params_direct,input_xyz,target_xyz)

        self.mirror = True
        fits = opt.minimize(chi2, [0, 0, 0],
                               args = (input_xyz,target_xyz))
        params_mirror = fits.x
        chi2_mirror = chi2(params_mirror,input_xyz,target_xyz)

        if chi2_mirror < chi2_direct :
            self.mirror = True
            self.angle = params_mirror[0]
            self.t[:2]  = params_mirror[1:3]
        else :
            self.mirror = False
            self.angle = params_direct[0]
            self.t[:2]  = params_direct[1:3]

        # compute rms distance
        transformed_xyz = self.apply(input_xyz)
        rms = np.sqrt(np.sum((target_xyz-transformed_xyz)**2)/target_xyz.shape[1])
        return rms

    def __str__(self) :
        r2d=180/np.pi
        line="Transfo2D: rotation angle (z) = {:.1f} deg\n".format(r2d*self.angle)
        line+="Transfo2D: translation dx={:.3f} dy={:.3f}\n".format(self.t[0],self.t[1])
        line+="Transfo2D: mirror = {}".format(self.mirror)
        return line

    @classmethod
    def read_dict(cls,params):
        tx = cls()
        tx.angle = params['a']
        tx.t[0] = params['dx']
        tx.t[1] = params['dy']
        tx.t[2] = 0
        tx.mirror = params['mirror']
        return tx

    def write_dict(self):
        params = dict()
        params['a'] = float(self.angle)
        params['dx'] = float(self.t[0])
        params['dy'] = float(self.t[1])
        params['mirror'] = bool(self.mirror)
        return params

    @classmethod
    def read_json(cls,filename):
        import json
        with open(filename) as fx:
            s = fx.read()
        tx = cls()
        params  = json.loads(s)
        return cls.read_dict(params)

    @classmethod
    def read_yaml(cls,filename):
        import yaml
        with open(filename) as fx:
            s = fx.read()
        params  = yaml.safe_load(s)
        return cls.read_dict(params)

    @classmethod
    def read(cls,filename):
        if filename.endswith("yaml"):
            return cls.read_yaml(filename)
        elif filename.endswith("json"):
            return cls.read_json(filename)
        else :
            raise IOError("Cannot write transfo to '{}'. Only yaml or json are supported.".format(filename))

    def write_json(self, filename):
        import json
        params = self.write_dict()
        with open(filename, 'w') as fx:
            fx.write(json.dumps(params))

    def write_yaml(self, filename):
        import yaml
        params = self.write_dict()
        with open(filename, 'w') as fx:
            fx.write(yaml.dump(params))

    def write(self, filename):
        if filename.endswith("yaml"):
            self.write_yaml(filename)
        elif filename.endswith("json"):
            self.write_json(filename)
        else :
            raise IOError("Cannot write transfo to '{}'. Only yaml or json are supported.".format(filename))

class OldTransfo3D() :
    '''
    Function to minimize when best fitting the transform matrix between input
    BMR locations and target BMR locations.

    mat: 1x6 Numpy array, with the first 3 values being a target rotation vector
    and the last 3 values a translation vector

    inputs: 3xN Numpy array containing measured BMR locations, where N is the
    number of Bmr

    targets: 3xN Numpy array containing target BMR locations
    '''

    # rotation then translation
    def __init__(self) :
        self.mirror    = False
        self.startvec  = np.array([0, 0, 1])
        self.targetvec = np.array([0, 0, 1])
        self.transvec = np.zeros(3)

    def rotmat3d(self,vec1, vec2):
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        if s == 0:
            rotation_matrix = np.eye(3)
        else:
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def rotmat(self) :
        return self.rotmat3d(self.startvec,self.targetvec)

    def apply(self,xyz) :

        xyz = np.atleast_1d(xyz)
        assert(xyz.shape[0]==3)

        res = xyz.copy()
        res = self.rotmat().dot(res)
        if self.mirror :
            res[0] *= -1
        if len(xyz.shape)==1 :
            res += self.transvec
        else :
            res += self.transvec[:,None]

        return res

    def fit(self,input_xyz,target_xyz) :

        def chi2(params, input_xyz, target_xyz) :
            self.targetvec = params[0:3]
            self.transvec  = params[3:6]
            transformed_xyz = self.apply(input_xyz)
            return np.sum ( (target_xyz-transformed_xyz)**2 )

        self.mirror = False
        fits = opt.minimize(chi2, [0, 0, 1, 0, 0, 0],
                               args = (input_xyz,target_xyz))
        params_direct = fits.x
        chi2_direct = chi2(params_direct,input_xyz,target_xyz)

        self.mirror = True
        fits = opt.minimize(chi2, [0, 0, 1, 0, 0, 0],
                               args = (input_xyz,target_xyz))
        params_mirror = fits.x
        chi2_mirror = chi2(params_mirror,input_xyz,target_xyz)

        if chi2_mirror < chi2_direct :
            self.mirror = True
            self.targetvec = params_mirror[0:3]
            self.transvec  = params_mirror[3:6]
        else :
            self.mirror = False
            self.targetvec = params_direct[0:3]
            self.transvec  = params_direct[3:6]

        # compute rms distance
        transformed_xyz = self.apply(input_xyz)
        rms = np.sqrt(np.sum((target_xyz-transformed_xyz)**2)/target_xyz.shape[1])
        return rms

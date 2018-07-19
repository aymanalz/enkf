import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg.blas
import dask.array as da
from dask import delayed
import time
class EnsKF(object):
    def __init__(self, Af=None, cf = None):
        self.Af = Af
        self.Cf = cf
        self.H = 1 # should be a vector of observed data
        self.rseed = None
        self.dseed = None
        self.add_error_to_inovation_mat = False
        self.compute_R_from_ensemble = False
        self.err_is_perc = True
        self.chunk_zise = 500
        self.obs_error = 1e-3
        self.trunc_method = 'eign_perc' # eign_num
        self.trunc_value = 1e-3
        self.states = None
        self.parameters = None
        self.observations = None
        self.hdf = None
        pass

    def update_parallel(self):
        """

        :param H: is the observable ensemble
        :param K: is the parameter  ensemble
        :param d:
        :param eps:
        :param h5_object:
        :param chunk_zise:
        :param cutoff:
        :param h_log:
        :return:
        """
        H = da.from_array(self.states, chunks=(self.chunk_zise, self.chunk_zise))
        K = da.from_array(self.parameters, chunks=(self.chunk_zise, self.chunk_zise))
        d = da.from_array(self.observations, chunks=(self.chunk_zise))
        h5_object = self.hdf
        m, N = H.shape
        nn, N = K.shape

        # Compute deviation of model predictions around ensemble mean
        h_dash = H - np.mean(H, axis=1).reshape((m, 1))
        if 0:
            if not h5_object == None:
                try:
                    h5_object.__delitem__('h_dash')
                except:
                    pass
                h5_object.create_dataset('h_dash', data =h_dash, dtype = 'float64' )
                h_dash = h5_object['h_dash']

        #Compute diagonal error R
        eps = self.obs_error
        if self.err_is_perc:
            #if self.rseed:
            #    np.random.seed(self.rseed)
            R = 0.0 + (eps/100.0) * np.std(H, axis=1)
            R =  da.from_array(R, chunks = self.chunk_zise)
            R = da.diag(R)
        else:
            # eps is error vector of the same dimension as observation
            if len(eps) != m:
                print " Error, number of measurement errors is not equal to obs. number  .... "
            R = da.from_array(np.array(eps), chunks=self.chunk_zise)
            R = da.diag(R)


        # Compute predicted observation covariance
        Chh = h_dash.dot(h_dash.transpose()) / (N - 1)
        Chh = Chh + R

        # compute pinverse
        if self.trunc_method == 'eign_perc':
            #TODO: use dask here
            Cinv, eig_val = self.pinv2(Chh, self.trunc_value)
        elif self.trunc_method == 'eign_num':
            Cinv, eig_val = self.pinv_numeig_dask(Chh, self.trunc_value)
        Cinv = da.from_array(Cinv, chunks=(self.chunk_zise, self.chunk_zise))

        #Cinv, eig_val = self.pinv2(Chh, self.cutoff_percent)
        if 0:
            if not h5_object== None:
                try:
                    h5_object.__delitem__('Cinv')
                except:
                    pass
                try:
                    h5_object.__delitem__('Eign_val')
                except:
                    pass
                h5_object.create_dataset('Cinv', data=Cinv)
                h5_object.create_dataset('Eign_val', data=eig_val)
                Cinv = h5_object['Cinv']

        # free memory
        Chh = None

        # compute inovation; inovation is deviation of model prediction from obsevation
        if self.add_error_to_inovation_mat:
            if self.dseed:
                np.random.seed(self.dseed)
            d_dash = d.value.reshape(m, 1) + (eps / 100.0) * np.random.randn(m, N) - H
        else:
            d_dash = d.reshape(m, 1) - H
        if 0:
            if not h5_object == None:
                try:
                    h5_object.__delitem__('d_dash')
                except:
                    pass
                h5_object.create_dataset('d_dash', data = d_dash)
                d_dash = h5_object['d_dash']

        # flush memory
        #h5_object.flush()
        end_of_ensemble = 1.0
        row_start = 0
        #k_a = h5_object['k_update']

        # compute update
        K_s_dash = K - K.mean(axis = 1).reshape((nn,1))
        Chk = K_s_dash.dot(h_dash.T)/ (N - 1)
        Chk_dot_Cinv = Chk.dot(Cinv)
        del_k = Chk_dot_Cinv.dot(d_dash)
        k_a = K + del_k
        start_time = time.time()
        #k_a.to_hdf5(self.hdf, 'update')
        k_a.to_hdf5(self.hdf.filename, 'kk_update', compression='lzf', shuffle=True)
        print("--- %s seconds ---" % (time.time() - start_time))


        if 0:
            chunk_zise = self.chunk_zise
            while end_of_ensemble:
                print "Percent Finished is {}".format(100.0*row_start/float(nn))
                if nn > row_start + chunk_zise:
                    row_end = row_start + chunk_zise
                else:
                    row_end = nn
                    end_of_ensemble = 0

                k_s = K[row_start:row_end,:]
                n,N = k_s.shape
                k_s_dash = k_s - np.mean(k_s, axis=1).reshape((n, 1))
                hdash = h_dash.value.transpose()

                #  Dot prodcut
                Chk = scipy.linalg.blas.dgemm(alpha=1.0, a=k_s_dash, b=hdash)/ (N - 1)
                Chk_dot_Cinv = scipy.linalg.blas.dgemm(alpha=1.0, a=Chk, b=Cinv)
                Chk = None
                del_k = scipy.linalg.blas.dgemm(alpha=1.0, a=Chk_dot_Cinv, b=d_dash)
                k_a[row_start:row_end,:] = k_s + del_k
                h5_object.flush()
                row_start = row_end
            pass




    def update(self):
        """

        :param H: is the observable ensemble
        :param K: is the parameter  ensemble
        :param d:
        :param eps:
        :param h5_object:
        :param chunk_zise:
        :param cutoff:
        :param h_log:
        :return:
        """
        H = self.states
        K = self.parameters
        d = self.observations
        h5_object = self.hdf
        m, N = H.shape
        nn, N = K.shape

        # Compute deviation of model predictions around ensemble mean
        h_dash = H - np.mean(H, axis=1).reshape((m, 1))
        if not h5_object == None:
            try:
                h5_object.__delitem__('h_dash')
            except:
                pass
            h5_object.create_dataset('h_dash', data =h_dash, dtype = 'float64' )
            h_dash = h5_object['h_dash']

        #Compute diagonal error R
        eps = self.obs_error
        if self.err_is_perc:
            #if self.rseed:
            #    np.random.seed(self.rseed)
            R = 0.0 + (eps/100.0) * np.std(H, axis=1)
        else:
            # eps is error vector of the same dimension as observation
            if len(eps) != m:
                print " Error, number of measurement errors is not equal to obs. number  .... "
            R = np.diag(eps)

        # Compute predicted observation covariance
        Chh = np.dot(h_dash.value, h_dash.value.transpose()) / (N-1)
        ind = np.diag_indices_from(Chh)
        if R.ndim == 2:
            Chh[ind] = Chh[ind] + R[ind]
        elif R.ndim == 1:
            Chh[ind] = Chh[ind] + R
        else:
            raise RuntimeError, "Dimension Error...."


        # compute pinverse
        if self.trunc_method == 'eign_perc':
            Cinv, eig_val = self.pinv2(Chh, self.trunc_value)
        elif self.trunc_method == 'eign_num':
            Cinv, eig_val = self.pinv_numeig(Chh, self.trunc_value)

        #Cinv, eig_val = self.pinv2(Chh, self.cutoff_percent)
        if not h5_object== None:
            try:
                h5_object.__delitem__('Cinv')
            except:
                pass
            try:
                h5_object.__delitem__('Eign_val')
            except:
                pass
            h5_object.create_dataset('Cinv', data=Cinv)
            h5_object.create_dataset('Eign_val', data=eig_val)
            Cinv = h5_object['Cinv']

        # free memory
        Chh = None

        # compute inovation; inovation is deviation of model prediction from obsevation
        if self.add_error_to_inovation_mat:
            if self.dseed:
                np.random.seed(self.dseed)
            d_dash = d.value.reshape(m, 1) + (eps / 100.0) * np.random.randn(m, N) - H
        else:
            d_dash = d.value.reshape(m, 1) - H

        if not h5_object == None:
            try:
                h5_object.__delitem__('d_dash')
            except:
                pass
            h5_object.create_dataset('d_dash', data = d_dash)
            d_dash = h5_object['d_dash']

        # flush memory
        h5_object.flush()
        end_of_ensemble = 1.0
        row_start = 0
        k_a = h5_object['k_update']

        # compute update
        chunk_zise = self.chunk_zise
        while end_of_ensemble:
            print "Percent Finished is {}".format(100.0*row_start/float(nn))
            if nn > row_start + chunk_zise:
                row_end = row_start + chunk_zise
            else:
                row_end = nn
                end_of_ensemble = 0

            k_s = K[row_start:row_end,:]
            n,N = k_s.shape
            k_s_dash = k_s - np.mean(k_s, axis=1).reshape((n, 1))
            hdash = h_dash.value.transpose()

            #  Dot prodcut
            Chk = scipy.linalg.blas.dgemm(alpha=1.0, a=k_s_dash, b=hdash)/ (N - 1)
            Chk_dot_Cinv = scipy.linalg.blas.dgemm(alpha=1.0, a=Chk, b=Cinv)
            Chk = None
            del_k = scipy.linalg.blas.dgemm(alpha=1.0, a=Chk_dot_Cinv, b=d_dash)
            k_a[row_start:row_end,:] = k_s + del_k
            h5_object.flush()
            row_start = row_end
        pass



    def update_0(self, A, D, H,  err_perc, cutoff):

        n, N = A.shape()
        D_dash = D - np.dot(H,A)
        N_1 = (1.0/N) * np.ones((N,N),dtype=float)
        I = np.eye(N,N)
        A_dash = np.dot(A,(I-N_1))
        S = np.dot(H,A)
        del(A_dash, H)
        m, N = np.shape(S)
        C = np.dot(S,S)
        errVar = C/(N-1)
        errVar = np.diag(np.diag(errVar))
        ErrSTD = np.power(errVar,0.5)
        ErrSTD = ErrSTD * err_perc
        D_dash = D_dash + np.diag(ErrSTD)
        C = C + ErrSTD
        del(Ce)
        if self.trunc_method == 'eign_perc':
            C_inv, eig_val = self.pinv2(C, self.trunc_value)
        elif self.trunc_method == 'eign_num':
            C_inv, eig_val = self.pinv_numeig(C, self.trunc_value)

        X = I + np.dot(S.transpose(),np.dot(C_inv,D_dash))
        A_U = np.dot(A , X)


    def pinv2(self, a, cut_percentage):

        a = a.conjugate()
        u, s, vt = np.linalg.svd(a, 0)
        eig_val = np.copy(s)
        m = u.shape[0]
        n = vt.shape[1]
        s_sum = np.sum(s)
        cutoff = s_sum * cut_percentage/100.0

        for i in range(np.min([n, m])):
            if s[i] > cutoff:
                s[i] = 1. / s[i]
            else:
                s[i] = 0.
        res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis], np.transpose(u)))

        return res, eig_val

    def pinv_numeig_dask(self, a, num_eig):
        """Cut the lowest number of eigne values"""

        print("Inverting the measurement covariance matrix")
        a = a.conj()
        u, s, vt = np.linalg.svd(a, 0)
        eig_val = np.copy(s)

        neig = int((len(s)*num_eig)/100.0)

        s = 1.0/s
        # force the last num_eig values to be zeros
        s[neig:] = 0.0
        res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis], np.transpose(u)))

        return res, eig_val

    def pinv_numeig(self, a, num_eig):
        """Cut the lowest number of eigne values"""

        a = a.conjugate()
        u, s, vt = np.linalg.svd(a, 0)
        eig_val = np.copy(s)

        neig = int((len(s)*num_eig)/100.0)

        s = 1.0/s
        # force the last num_eig values to be zeros
        s[neig:] = 0.0
        res = np.dot(np.transpose(vt), np.multiply(s[:, np.newaxis], np.transpose(u)))

        return res, eig_val



if __name__=="__main__":


    pass


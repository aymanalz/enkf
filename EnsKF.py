__version__ = 0.0
__Date__ = "7/25/2018"


import numpy as np
import scipy.linalg.blas
import dask.array as da
from dask import delayed
import time

class EnsKF(object):
    """
    Ensemble Kalman Filter
    ----------------------
    A class to implement to the update stage for Ensemble Kalman Filter, Ensemble Smoother, and Ensemble Kalman
    Smoother.

    Mathematical Notations:
    * Consider a model that accepts K as an input and produces H as an output  H = G(K)
    * An esemble of K's is used to produce an esemble of H's.
    * The forcast matrix (Af) can be formed as Af = [[Hs],[Ks]]


    ## Illustrative Example
    -----------
    import EnKF
    kf = EnKF()

    # The hdf database "hdf_file.h5" contains the
    kf.hdf = h5py.File('Monte_carlo_simulation_dataset.h5', 'a')
    kf.observations = hdf.get('obs')  # observation vector
    kf.parameters = hdf.get('k_ens')  # parameter ensemble
    kf.states = hdf.get('h_ens')      # states ensemble
    kf.obs_error = eps                # Observation Error vector
    kf.err_is_perc = False            # How error obs_error should be interpreted; either percentage of the prior
                                        variance  or abolute error value
    kf.trunc_method = 'eign_num'      # define how TSVD is com[uted: either as percentage of the number of eigenvalues
                                       'eign_num', or percentation of the summation of eigenvalues
    kf.trunc_value = cut              # Truncation threshold value.
    kf.chunk_zise = 5000              # chunck size to pull from the hdf file

    # Implement the Update
    kf.update_parallel()

    # The posterior Ensemble is
    kk_update = hdf.get('kk_update')

    """
    def __init__(self, hdf = None, states = None, parameters = None, observations = None):
        """

        :param :
        hdf :       h5 file that contains the Monte Carlo Simulation results
        states:     A matrix (within hdf) that contains the model observable response
        parameters: A matrix that contains model parameters; for example hydraulic conductivity. Also the matrix may
                    contain any model responce that is not observed and needs to be estimated or predicted.
        rseed :     seed number for error generation --- TODO: not active
        dseed:      seed number for observation noise ---- TODO: not active

        :param :
        """
        self.states = states
        self.parameters = parameters
        self.observations = observations
        self.hdf = hdf
        self.rseed = 123
        self.dseed = 456
        self.add_error_to_innovation_mat = False
        self.compute_R_from_ensemble = False
        self.err_is_perc = True
        self.chunk_zise = 500
        self.obs_error = 1e-3
        self.trunc_method = 'eign_perc' # eign_num
        self.trunc_value = 1e-3
        self.save_intermediate_matrices = False

    def update_parallel(self):
        """
         parallelization is implemented by DASK
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
        if self.save_intermediate_matrices:  # TODO: Have not tested
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
            #    np.random.seed(self.rseed) #TODO: activate & test
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
        if self.save_intermediate_matrices:  # TODO: Have not tested
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
        if self.add_error_to_innovation_mat:
            if self.dseed:
                np.random.seed(self.dseed)
            d_dash = d.value.reshape(m, 1) + (eps / 100.0) * np.random.randn(m, N) - H
        else:
            d_dash = d.reshape(m, 1) - H
        if self.save_intermediate_matrices:  # TODO: Have not tested :
            if not h5_object == None:
                try:
                    h5_object.__delitem__('d_dash')
                except:
                    pass
                h5_object.create_dataset('d_dash', data = d_dash)
                d_dash = h5_object['d_dash']

        # flush memory
        if self.save_intermediate_matrices:  # TODO: Have not tested
            h5_object.flush()

        # compute update
        K_s_dash = K - K.mean(axis = 1).reshape((nn,1))
        Chk = K_s_dash.dot(h_dash.T)/ (N - 1)
        Chk_dot_Cinv = Chk.dot(Cinv)
        del_k = Chk_dot_Cinv.dot(d_dash)
        k_a = K + del_k

        # DASK will delay all calculation till this moment
        start_time = time.time()
        k_a.to_hdf5(self.hdf.filename, 'kk_update', compression='lzf', shuffle=True)
        print("--- %s seconds ---" % (time.time() - start_time))


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
        if self.add_error_to_innovation_mat:
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

    def pinv2(self, a, cut_percentage):
        """
        Compute the pseudoinverse of the observation covariance matrix
        :param a:
        :param cut_percentage:
        :return:
        """

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
        """
        Compute the pseudoinverse of the observation covariance matrix
        """

        print("Inverting the measurement covariance matrix.....")
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
        """
        Compute the pseudoinverse of the observation covariance matrix
        """

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


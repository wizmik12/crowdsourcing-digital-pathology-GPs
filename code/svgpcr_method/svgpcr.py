import numpy as np
import tensorflow as tf

from gpflow import kullback_leiblers, features
from gpflow import settings
from gpflow import transforms
from gpflow.conditionals import conditional, Kuu
from gpflow.decors import params_as_tensors
from gpflow.models.model import GPModel
from gpflow.params import DataHolder, Minibatch, Parameter

import time

class SVGPCR(GPModel):

    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None,
                 feat=None,
                 Z=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 num_data=None,
                 num_latent = None,
                 q_mu=None,
                 q_sqrt=None,
                 alpha = None,
                 alpha_tilde = None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y contains the annotations. It is a numpy array of matrices with 2 columns, gathering pairs (annotator, annotation).
        - kern, likelihood, mean_function are appropriate GPflow objects
        - feat and Z define the pseudo inputs, usually feat=None and Z size M x D
        - q_diag, boolean indicating whether posterior covariance must be diagonal
        - withen, boolean indicating whether a whitened representation of the inducing points is used
        - minibatch_size, if not None, turns on mini-batching with that size
        - num_data is the total number of observations, default to X.shape[0] (relevant when feeding in external minibatches)
        - num_latent is the number of latent GP to be used. For multi-class likelihoods, this equals the number of classes. However, for many binary likelihoods, num_latent=1.
        - q_mu (M x K), q_sqrt (M x K or K x M x M), alpha (A x K x K), alpha_tilde (A x K x K), initializations for these parameters (all of them but alpha to be estimated).
        """
        if minibatch_size is None:
            X = DataHolder(X)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
        class_keys = np.unique(np.concatenate([y[:,1] for y in Y]))
        num_classes = len(class_keys)
        num_latent = num_latent or num_classes
        GPModel.__init__(self, X, None, kern, likelihood, mean_function, num_latent, **kwargs)
        self.class_keys = class_keys
        self.num_classes = num_classes
        self.num_latent = num_latent
        self.annot_keys = np.unique(np.concatenate([y[:,0] for y in Y]))
        self.num_annotators = len(self.annot_keys)
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_inducing = len(self.feature)

        ###### Initializing Y_idxs as minibatch or placeholder (and the associated idxs to slice q_unn) ######################
        startTime = time.time()
        Y_idxs = np.array([np.stack((np.array([np.flatnonzero(v==self.annot_keys)[0] for v in y[:,0]]),
                                     np.array([np.flatnonzero(v==self.class_keys)[0] for v in y[:,1]])), axis=1) for y in Y]) # same as Y but with indexes
        S = np.max([v.shape[0] for v in Y_idxs])
        ###########################################
        ## pmr modification for CPU
        #Y_idxs_cr = np.array([np.concatenate((y,-1*np.ones((S-y.shape[0],2))),axis=0) for y in Y_idxs]).astype(np.int16) # NxSx2
        aux = np.array([self.num_annotators,0])
        Y_idxs_cr = np.array([np.concatenate((y,np.tile(aux,(S-y.shape[0],1))),axis=0) for y in Y_idxs]).astype(np.int16) # NxSx2
        ###########################################
        

        if minibatch_size is None:
            self.Y_idxs_cr = DataHolder(Y_idxs_cr)
            self.idxs_mb = DataHolder(np.arange(self.num_data))
        else:
            self.Y_idxs_cr = Minibatch(Y_idxs_cr, batch_size=minibatch_size, seed=0)
            self.idxs_mb = Minibatch(np.arange(self.num_data), batch_size=minibatch_size, seed=0)
        print("Time taken in Y_idxs creation:", time.time()-startTime)

        ########## Initializing q #####################################
        startTime = time.time()
        q_unn = np.array([np.bincount(y[:,1], minlength=self.num_classes) for y in Y_idxs])
        q_unn = q_unn + np.ones(q_unn.shape)
        q_unn = q_unn/np.sum(q_unn,axis=1,keepdims=True)
        self.q_unn = Parameter(q_unn,transform=transforms.positive) # N x K
        print("Time taken in q_unn initialization:", time.time()-startTime)

        ######## Initializing alpha (fix) and alpha_tilde (trainable) ################3
        #if alpha is None:
        #    self.alpha = tf.constant(np.ones((self.num_annotators,self.num_classes,self.num_classes), dtype=settings.float_type)) # A x K x K
        #else:
        #    self.alpha = tf.constant(alpha, dtype=settings.float_type) # A x K x K

        if alpha is None:
            alpha = np.ones((self.num_annotators,self.num_classes,self.num_classes), dtype=settings.float_type) # A x K x K
        self.alpha = Parameter(alpha, transform=transforms.positive, trainable=False)

        startTime = time.time()
        alpha_tilde = self._init_behaviors(q_unn, Y_idxs)
        print("Time taken in alpha_tilde initialization:", time.time()-startTime)
        self.alpha_tilde = Parameter(alpha_tilde,transform=transforms.positive) # A x K x K
        ################################################################################
        ##### Initializing the variational parameters  ####################################
        self._init_variational_parameters(q_mu, q_sqrt)

    def _init_variational_parameters(self, q_mu, q_sqrt):
        q_mu = np.zeros((self.num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x K

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((self.num_inducing, self.num_latent), dtype=settings.float_type), transform=transforms.positive)  # M x K
            else:
                q_sqrt = np.array([np.eye(self.num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(self.num_inducing, self.num_latent))  # K x M x M
        else:
            if self.q_diag:
                assert q_sqrt.ndim == 2
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x K
            else:
                assert q_sqrt.ndim == 3
                self.q_sqrt = Parameter(q_sqrt,transform=transforms.LowerTriangular(self.num_inducing, self.num_classes))  # K x M x M

    def _init_behaviors(self, probs, Y_idxs):
        alpha_tilde = np.ones((self.annot_keys.size,self.class_keys.size,self.class_keys.size))/self.class_keys.size
        counts = np.ones((self.annot_keys.size,self.class_keys.size))
        print(len(Y_idxs))
        for n in range(len(Y_idxs)):
            for a,c in zip(Y_idxs[n][:,0], Y_idxs[n][:,1]):
                alpha_tilde[a,c,:] += probs[n,:]
                counts[a,c] += 1
        alpha_tilde=alpha_tilde/counts[:,:,None]
        alpha_tilde = (counts/np.sum(counts,axis=1,keepdims=True))[:,:,None]*alpha_tilde
        return alpha_tilde/np.sum(alpha_tilde,axis=1,keepdims=True)

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)
        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def build_annot_KL(self):
        alpha_diff = self.alpha_tilde-self.alpha
        KL_annot=(tf.reduce_sum(tf.multiply(alpha_diff,tf.digamma(self.alpha_tilde)))-
        tf.reduce_sum(tf.digamma(tf.reduce_sum(self.alpha_tilde,1))*tf.reduce_sum(alpha_diff,1))+
        tf.reduce_sum(tf.lbeta(tf.matrix_transpose(self.alpha))
        -tf.lbeta(tf.matrix_transpose(self.alpha_tilde))))
        return KL_annot

    @params_as_tensors
    def _build_likelihood(self):

        KL = self.build_prior_KL()
        KL_annot = self.build_annot_KL()

        ########################## ENTROPY OF Q COMPONENT #############################
        q_unn_mb = tf.gather(self.q_unn,self.idxs_mb)   # N x K
        q_mb = tf.divide(q_unn_mb, tf.reduce_sum(q_unn_mb, axis=1, keepdims=True)) # N x K
        qentComp = tf.reduce_sum(tf.multiply(q_mb,tf.log(q_mb)))

        ######################### LIKELIHOOD COMPONENT ################################
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)
        tensors_list = [self.likelihood.variational_expectations(fmean, fvar, c*tf.ones((tf.shape(self.X)[0],1),dtype=tf.int32)) for c in np.arange(self.num_classes)]
        tnsr_lik = tf.concat(tensors_list,-1)  # NxK
        lhoodComp = tf.reduce_sum(tf.multiply(q_mb,tnsr_lik))

        ######################### CROWDSOURCING COMPONENT ################################
        expect_log = tf.digamma(self.alpha_tilde)-tf.digamma(tf.reduce_sum(self.alpha_tilde,1,keepdims=True)) # A x K x K
        
        ####################################
        # prm modification for CPU
        expect_log = tf.concat([expect_log,tf.zeros([1,self.num_classes,self.num_classes],tf.float64)],0)  
        ####################################
        
        tnsr_expCrow = tf.gather_nd(expect_log, tf.cast(self.Y_idxs_cr, tf.int32)) # N x S x K (on GPU, indexes -1 return 0, see documentation for tf.gather_nd. On CPU, an error would be raised.) <---- Now it shouldn't be true
        crComp = tf.reduce_sum(tf.multiply(tnsr_expCrow, tf.expand_dims(q_mb,1)))

        scale = tf.cast(self.num_data, settings.float_type)/tf.cast(tf.shape(self.X)[0], settings.float_type)
        self.decomp = [lhoodComp,crComp,qentComp,KL,KL_annot,scale]
        return ((lhoodComp+crComp-qentComp)*scale-KL-KL_annot)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(Xnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov, white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

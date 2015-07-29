from __future__ import division
import numpy as np
import time
import re
import os
import operator
import cPickle as pickle
import dateutil
from urllib2 import urlopen
from collections import namedtuple

from pybasicbayes.util.text import progprint, progprint_xrange

from ctm import split_test_train
from pgmult.lda import *

#############
#  models   #
#############

class ElectronicHealthRecordDTM(StickbreakingDynamicTopicsLDA):
    """
    Extension of the dynamic topic model (DTM) with extra
    variables specific to modeling electronic health records.
    Specifically, we augment the data with the following fields:
    - patient_id: the id of the patient that each document belongs to

    All the documents for a given patient share the same Dirichlet params
    """
    def __init__(self, data, timestamps, K, alpha_theta, patient_ids, lda_model=None):
        super(ElectronicHealthRecordDTM, self).\
            __init__(data, timestamps, K, alpha_theta, lda_model=lda_model)

        assert patient_ids.shape == (self.D,)
        self.patient_ids = patient_ids
        self.unique_patient_ids = np.unique(patient_ids)

    def initialize_parameters(self, lda_model=None):
        """
        Initialize the model with
        :param lda_model:
        :return:
        """

        # TODO make this learned, init from hypers
        self.sigmasq_states = 0.1

        # Allocate auxiliary variables
        self.omega = np.zeros((self.T, self.V-1, self.K))

        # If LDA model is given, use it to initialize beta and theta
        if lda_model:
            assert isinstance(lda_model, _LDABase)
            assert lda_model.D == self.D
            assert lda_model.V == self.V
            assert lda_model.K == self.K

            self.beta = lda_model.beta

            # Share the topic distributions among documents
            # belonging to the same patient
            for pid in self.unique_patient_ids:
                patient_theta = \
                    np.mean(lda_model.theta[self.patient_ids==pid], axis=0)
                assert np.allclose(patient_theta.sum(), 1.0)
                self.theta[self.patient_ids==pid] = patient_theta

        else:
            # Initialize beta to uniform and theta from prior
            mean_psi = compute_uniform_mean_psi(self.V)[0][None,:,None]
            self.psi = np.tile(mean_psi, (self.T, 1, self.K))

            self.theta = np.zeros((self.D, self.K))
            for pid in self.unique_patient_ids:
                self.theta[self.patient_ids==pid] = \
                    np.random.dirichlet(self.alpha_theta * np.ones(self.K))

        # Sample topic-word assignments
        self.z = np.zeros((self.data.data.shape[0], self.K), dtype='uint32')
        self.resample_z()

    # Override theta resampling to share counts across docs for a given patient
    def resample_theta(self):
        dtc = self.doc_topic_counts
        for pid in self.patient_ids:
            alpha_post = \
                self.alpha_theta + dtc[self.patient_ids==pid].sum(axis=0)
            self.theta[self.patient_ids==pid] = np.random.dirichlet(alpha_post)



class ForumAndEHRDTM(ElectronicHealthRecordDTM):
    """
    Further extension to handle Forum and EHR documents
    - doc_type:   the type of document, EHR or forum post

    Each document type has its own drifting betas, but the mean
    of the betas is shared for forums and EHRs
    """
    def __init__(self, data, timestamps, K, alpha_theta, patient_ids, doc_types, lda_model=None):
        super(ForumAndEHRDTM, self).\
            __init__(data, timestamps, K, alpha_theta, patient_ids, lda_model=lda_model)

        # Doc types must be a binary array
        # doc_type[i] == 0 implies document is EHR
        # doc_type[i] == 1 implies document is Forum
        assert doc_types.shape == (self.D,) and np.dtype(doc_types) == np.int\
               and np.amin(doc_types) == 0 and np.amax(doc_types) == 1
        self.doc_types = doc_types

    @property
    def beta(self):
        return np.concatenate((self.beta_ehr[None,:,:,:],
                               self.beta_forum[None,:,:,:]),
                              axis=0)

    @property
    def beta_ehr(self):
        return psi_to_pi(self.psi_ehr, axis=1)

    @property
    def beta_forum(self):
        return psi_to_pi(self.psi_forum, axis=1)

    def resample_u(self):
        # Resample the latent states that govern the mean of beta's
        mu_init, sigma_init, sigma_states, sigma_obs, y = \
            self._get_lds_effective_params()
        _, u_flat = filter_and_sample_randomwalk(
            mu_init, sigma_init, sigma_states, sigma_obs, y)
        self.u = u_flat.reshape(self.u.shape)

    def resample_psi(self):
        # Get the prior
        mu_prior = self.u
        sigmasq_prior = self.sigmasq_obs


        # Resample psi_ehr
        sigmasq_obs_ehr = 1./self.omega_ehr
        mu_obs_ehr = kappa_vec(self.ehr_time_word_topic_counts, axis=1) / self.omega_ehr

        sigmasq_post_ehr = 1./(1./sigmasq_prior + 1./sigmasq_obs_ehr)
        mu_post_ehr = sigmasq_post_ehr * (mu_prior/sigmasq_prior + mu_obs_ehr/sigmasq_obs_ehr)
        self.psi_ehr = mu_post_ehr + np.sqrt(sigmasq_post_ehr) * np.random.randn(self.T, self.V-1, self.K)

        # Resample psi_forum
        sigmasq_obs_forum = 1./self.omega_forum
        mu_obs_forum = kappa_vec(self.forum_time_word_topic_counts, axis=1) / self.omega_forum

        sigmasq_post_forum = 1./(1./sigmasq_prior + 1./sigmasq_obs_forum)
        mu_post_forum = sigmasq_post_forum * (mu_prior/sigmasq_prior + mu_obs_forum/sigmasq_obs_forum)
        self.psi_forum = mu_post_forum + np.sqrt(sigmasq_post_forum) * np.random.randn(self.T, self.V-1, self.K)

    def resample_omega(self):
        pgdrawvpar(
            self.ppgs,
            N_vec(self.ehr_time_word_topic_counts, axis=1)
                .astype('float64').ravel(),
            self.psi_ehr.ravel(), self.omega_ehr.ravel())
        np.clip(self.omega_ehr, 1e-32, np.inf, out=self.omega_ehr)

        pgdrawvpar(
            self.ppgs,
            N_vec(self.forum_time_word_topic_counts, axis=1)
                .astype('float64').ravel(),
            self.psi_forum.ravel(), self.omega_forum.ravel())
        np.clip(self.omega_forum, 1e-32, np.inf, out=self.omega_forum)


    def _get_lds_effective_params(self):
        mu_uniform, sigma_uniform = compute_uniform_mean_psi(self.V)
        mu_init = np.tile(mu_uniform, self.K)
        sigma_init = np.tile(np.diag(sigma_uniform), self.K)

        sigma_states = np.repeat(self.sigmasq_states, (self.V - 1) * self.K)

        # Observations are psi_ehr and psi_forum. Both are N(u, \sigma_obs^2)
        # p(psi_ehr, psi_forum | u) = p(psi_ehr \ u) p(psi_forum | u)
        # = N(psi_ehr | u, sig) N(psi_forum | u, sig)
        # = -0.5/sig^2 * [(psi_ehr - u)^2 + (psi_forum -u)^2]
        # = -0.5/sig^2 * [(psi_ehr^2 + psi_forum^2) -2(psi_ehr + psi_forum)u + 2u^2]
        # = -0.5*2/sig^2 * [(psi_ehr^2 + psi_forum^2)/2 -(psi_ehr + psi_forum)/2 u + u^2]
        # = N((psi_ehr + psi_forum)/2 | u, sig / \sqrt{2})
        sigma_obs = self.sigmasq_obs / 2.
        y = (self.psi_ehr + self.psi_forum) / 2.

        return mu_init, sigma_init, sigma_states, \
            sigma_obs.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1)


    def _update_counts(self):
        self.doc_topic_counts = np.zeros((self.D, self.K), dtype='uint32')
        self.time_word_topic_counts = None
        self.ehr_time_word_topic_counts = np.zeros((self.T, self.V, self.K), dtype='uint32')
        self.forum_time_word_topic_counts = np.zeros((self.T, self.V, self.K), dtype='uint32')
        rows, cols = csr_nonzero(self.data)
        row_doc_types = self.doc_types[rows]
        for dt, i, j, t, zvec in zip(row_doc_types, rows, cols, self.timeidx, self.z):
            self.doc_topic_counts[i] += zvec
            if dt == 0:
                self.ehr_time_word_topic_counts[t,j] += zvec
            else:
                self.forum_time_word_topic_counts[t,j] += zvec

    def _get_topicprobs(self):
        rows, cols = csr_nonzero(self.data)
        return normalize_rows(self.theta[rows] * self.beta[self.doc_types, self.timeidx, cols])

    def _get_wordprobs(self, data, timeidx):
        rows, cols = csr_nonzero(data)
        return np.einsum('tk,tk->t',self.theta[rows],self.beta[self.doc_types, timeidx, cols])


#############
#  fitting  #
#############

Results = namedtuple(
    'Results', ['loglikes', 'predictive_lls', 'samples', 'timestamps'])


def fit_sbdtm_gibbs(train_data, test_data, timestamps, K, Niter, alpha_theta):
    def evaluate(model):
        ll, pll = \
            model.log_likelihood(), \
            model.log_likelihood(test_data)
        # print '{} '.format(ll),
        return ll, pll

    def sample(model):
        tic = time.time()
        model.resample()
        timestep = time.time() - tic
        return evaluate(model), timestep

    print 'Running sbdtm gibbs...'
    model = StickbreakingDynamicTopicsLDA(train_data, timestamps, K, alpha_theta)
    init_val = evaluate(model)
    vals, timesteps = zip(*[sample(model) for _ in progprint_xrange(Niter)])

    lls, plls = zip(*((init_val,) + vals))
    times = np.cumsum((0,) + timesteps)

    return Results(lls, plls, model.copy_sample(), times)


#############
#  running  #
#############

if __name__ == '__main__':
    ## sotu
    # K, V = 25, 2500  # TODO put back
    K, V = 5, 100
    alpha_theta = 1.
    train_frac, test_frac = 0.95, 0.5

    # TODO: Load EHR and forum data
    timestamps, (data, words), patient_ids, doc_types = load_data()

    ## print setup
    print 'K=%d, V=%d' % (K, V)
    print 'alpha_theta = %0.3f' % alpha_theta
    print 'train_frac = %0.3f, test_frac = %0.3f' % (train_frac, test_frac)
    print

    ## split train test
    train_data, test_data = split_test_train(data, train_frac=train_frac, test_frac=test_frac)

    ## fit
    sb_results = fit_sbdtm_gibbs(train_data, test_data, timestamps, K, 100, alpha_theta)

    all_results = {
        'sb': sb_results,
    }

    with open('dtm_results.pkl','w') as outfile:
        pickle.dump(all_results, outfile, protocol=-1)


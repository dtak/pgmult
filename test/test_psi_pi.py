import numpy as np
np.random.seed(2)

import matplotlib.pyplot as plt

from pgmult.distributions import PGMultinomial
from pgmult.internals.utils import compute_uniform_mean_psi, pi_to_psi, psi_to_pi

from pybasicbayes.util.text import progprint_xrange

def test_psi_pi_conversion():
    K = 10000

    pi = np.random.dirichlet(0.1 * np.ones(K))
    assert np.all(np.isfinite(pi))
    # pi = np.ones(K) / float(K)
    psi = pi_to_psi(pi)
    pi2 = psi_to_pi(psi)

    # print "pi:  ", pi
    # print "psi: ", psi
    # print "pi2: ", pi2

    assert np.allclose(pi, pi2), "Mapping is not invertible."


def test_psi_pi_conversion_2d():
    K = 10000
    M = 100

    pi = np.random.dirichlet(0.1 * np.ones(K), size=M)
    assert np.all(np.isfinite(pi))

    psi_a = pi_to_psi(pi)
    psi_b = pi_to_psi(pi, axis=1)
    assert np.allclose(psi_a, psi_b)

    pi_a = psi_to_pi(psi_a)
    pi_b = psi_to_pi(psi_b, axis=1)
    assert np.allclose(pi_a, pi_b, atol=1e-7)
    assert np.allclose(pi_a, pi, atol=1e-7)


def test_psi_pi_conversion_3d():
    K = 10
    M = 10

    pi = np.random.dirichlet(0.1 * np.ones(K), size=(M,M))
    assert pi.shape == (M,M,K)
    assert np.all(np.isfinite(pi))

    psi = pi_to_psi(pi, axis=2)
    pi2 = psi_to_pi(psi, axis=2)
    assert np.allclose(pi2, pi, atol=1e-7)

def test_pgm_rvs():
    K = 10
    mu, sig = compute_uniform_mean_psi(K, sigma=2)
    # mu = np.zeros(K-1)
    # sig = np.ones(K-1)
    print "mu:  ", mu
    print "sig: ", sig

    Sigma = np.diag(sig)

    # Add some covariance
    # Sigma[:5,:5] = 1.0 + 1e-3*np.random.randn(5,5)

    # Sample a bunch of pis and look at the marginals
    pgm = PGMultinomial(K, mu=mu, Sigma=Sigma)
    samples = 10000
    pis = []
    for smpl in xrange(samples):
        pgm.resample()
        pis.append(pgm.pi)
    pis = np.array(pis)

    print "E[pi]:   ", pis.mean(axis=0)
    print "var[pi]: ", pis.var(axis=0)

    plt.figure()
    plt.subplot(121)
    plt.boxplot(pis)
    plt.xlabel("k")
    plt.ylabel("$p(\pi_k)$")

    # Plot the covariance
    cov = np.cov(pis.T)
    plt.subplot(122)
    plt.imshow(cov, interpolation="None", cmap="cool")
    plt.colorbar()
    plt.title("Cov($\pi$)")
    plt.show()


def test_correlated_pgm_rvs(Sigma):
    K = Sigma.shape[0] + 1
    mu, _ = compute_uniform_mean_psi(K)
    print "mu:  ", mu

    # Sample a bunch of pis and look at the marginals
    samples = 10000
    psis = np.random.multivariate_normal(mu, Sigma, size=samples)
    pis = []
    for smpl in xrange(samples):
        pis.append(psi_to_pi(psis[smpl]))
    pis = np.array(pis)

    print "E[pi]:   ", pis.mean(axis=0)
    print "var[pi]: ", pis.var(axis=0)

    plt.figure()
    plt.subplot(311)
    plt.boxplot(pis)
    plt.xlabel("k")
    plt.ylabel("$p(\pi_k)$")

    # Plot the covariance
    cov = np.cov(pis.T)
    plt.subplot(323)
    plt.imshow(cov[:-1,:-1], interpolation="None", cmap="cool")
    plt.colorbar()
    plt.title("Cov($\pi$)")

    plt.subplot(324)
    invcov = np.linalg.inv(cov[:-1,:-1] + np.diag(1e-6 * np.ones(K-1)))
    # good = np.delete(np.arange(K), np.arange(0,K,3))
    # invcov = np.linalg.inv(cov[np.ix_(good,good)])
    plt.imshow(invcov, interpolation="None", cmap="cool")
    plt.colorbar()
    plt.title("Cov$(\pi)^{-1}$")

    plt.subplot(325)
    plt.imshow(Sigma, interpolation="None", cmap="cool")
    plt.colorbar()
    plt.title("$\Sigma$")

    plt.subplot(326)
    plt.imshow(np.linalg.inv(Sigma), interpolation="None", cmap="cool")
    plt.colorbar()
    plt.title("$\Sigma^{-1}$")


    plt.savefig("correlated_psi_pi.png")
    plt.show()

def test_chain_correlated_pgm_rvs(K=10):
    Sigma = np.linalg.inv(np.eye(K) + np.diag(np.repeat(0.5,K-1),k=1) + np.diag(np.repeat(0.5,K-1),k=-1))
    test_correlated_pgm_rvs(Sigma)

def test_wishart_correlated_pgm_rvs(K=10):
    # Randomly generate a covariance matrix
    from pybasicbayes.util.stats import sample_invwishart
    Sigma = sample_invwishart(np.eye(K-1), nu=K)
    test_correlated_pgm_rvs(Sigma)

def test_block_correlated_pgm_rvs():
    n = 3
    Sblocks = 2.0 * np.eye(n) + np.diag(np.repeat(0.5,n-1),k=1) + np.diag(np.repeat(0.5,n-1),k=-1)
    Sigma = np.kron(Sblocks,np.eye(3))
    test_correlated_pgm_rvs(Sigma)

for itr in progprint_xrange(1000):
    # test_psi_pi_conversion()
    # test_psi_pi_conversion_2d()
    test_psi_pi_conversion_3d()

# test_pgm_rvs()
# test_chain_correlated_pgm_rvs()
# test_wishart_correlated_pgm_rvs(K=10)
# test_block_correlated_pgm_rvs()
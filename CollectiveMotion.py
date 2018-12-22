import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

np.random.seed(1234)
rng = np.random.RandomState(1234)
trng = RandomStreams(rng.randint(999999))

class CollectiveMotion:
    def __init__(self,
                 inf     = 1e37):

        pos, vel       = T.fmatrices(['pos', 'vel'])
        nc, N, n_steps = T.iscalars(['nc', 'N', 'n_steps'])
        ra, rb, re, r0 = T.fscalars(['ra', 'rb', 're', 'r0'])
        v0, j, b       = T.fscalars(['v0', 'J', 'b'])


        nu = trng.uniform(size=(N,2), low=0.0, high=3.14159, dtype='floatX')


        def distance_tensor(X):
                E = X.reshape((X.shape[0], 1, -1)) - X.reshape((1, X.shape[0], -1))
                D = T.sqrt(T.sum(T.square(E), axis=2))
                return D

        def direction_tensor(X):
            E = X.reshape((X.shape[0], 1, -1)) - X.reshape((1, X.shape[0], -1))
            L = T.sqrt(T.sum(T.square(E), axis=2))
            L = T.pow(L + T.identity_like(L), -1)
            L = T.stack([L, L, L], axis=2)
            return L * E

        def neighbourhood(X):
            D = distance_tensor(X)
            N = T.argsort(D, axis=0)
            mask = T.cast(T.lt(N,nc), 'float32')
            return N[1:nc+1], mask

        def alignment(X,Y):
            n, d = neighbourhood(X)
            return T.sum(Y[n], axis=0)

        def cohesion(X):
            D    = distance_tensor(X)
            E    = direction_tensor(X)
            n, d = neighbourhood(X)

            F = T.zeros_like(E)
            D = T.stack([D, D, D], axis=2)
            d = T.stack([d, d, d], axis=2)

            c1 = T.lt(D, rb)
            c2 = T.and_(T.gt(D, rb), T.lt(D, ra))
            c3 = T.and_(T.gt(D, ra), T.lt(D, r0))

            F = T.set_subtensor(F[c1], -E[c1])
            F = T.set_subtensor(F[c2], 0.25 * (D[c2] - re) / (ra - re) * E[c2])
            F = T.set_subtensor(F[c3], E[c3])

            return T.sum(d * F, axis=0)

        def perturbation(nu = nu):
            phi   = nu[:,0]
            theta = 2.0 * nu[:,1]

            return T.stack([T.sin(theta) * T.sin(phi), T.cos(theta) * T.sin(phi), T.cos(phi)], axis=1)

        def step(X,dX):
            X_ = X + dX
            V_ = j * nc / v0 * (alignment(X, dX)) + b * (cohesion(X)) + nc * (perturbation())
            dV = T.sqrt(T.sum(T.square(V_), axis=1)).reshape(V_.shape[0],1)
            dV = T.stack([dV, dV, dV], axis=1)
            V  = v0 * V_ / dV

            return T.cast(X_, 'float32'), T.cast(V, 'float32')

    
        sim, update = theano.scan(step,
                                  outputs_info=[pos,vel],
                                  n_steps=n_steps)

        self.f = theano.function([pos, vel, nc, ra, rb, r0, re, j, v0, b, N, n_steps],
                                  sim,
                                  allow_input_downcast=True,
                                  on_unused_input='ignore')


    def simulate_particles(self,
                           J       = 0.02,
                           N       = 256,
                           nc      = 20,
                           ra      = 0.8,
                           rb      = 0.2,
                           re      = 0.5,
                           r0      = 1.0,
                           v0      = 0.05,
                           b       = 5.0,
                           n_steps = 500):
        test_nu = np.random.uniform(0.0, np.pi, size=(N,2))
        test_r  = np.random.uniform(0.0, 1.0, size=(N,1))
        test_pos = test_r * np.column_stack([np.sin(2.0*test_nu[:,0]) * np.sin(test_nu[:,1]), 
                                            np.cos(2.0*test_nu[:,0]) * np.sin(test_nu[:,1]), 
                                            np.cos(test_nu[:,1])])
        test_vel = np.zeros((N,3))

        x, v = self.f(test_pos, test_vel, nc, ra, rb, r0, re, J, v0, b, N, n_steps)
        return x,v

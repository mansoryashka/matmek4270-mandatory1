import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        X = np.linspace(0, 1, N+1)
        Y = np.linspace(0, 1, N+1)
        self.h = X[1] - X[0]
        
        self.xij, self.yij = np.meshgrid(X, Y, sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')

        return D2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        w = self.c*np.pi*np.sqrt(self.mx**2 + self.my**2)
        # print(self.c*np.pi*np.sqrt(self.mx**2 + self.my**2))

        # kx = self.mx*np.pi
        # ky = self.my*np.pi
        # w = self.c*np.sqrt(kx**2 + ky**2)

        # w = 1.443
        return w
    
    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.Unm1 = np.zeros((N+1,N+1))
        self.Un = np.zeros((N+1,N+1))

        u0 = sp.lambdify((x, y), self.ue(mx, my).subs({t:0}))
        u1 = sp.lambdify((x, y), self.ue(mx, my).subs({t:self.dt}))

        self.Unm1[:] = u0(self.xij, self.yij)
        self.Un[:] = u1(self.xij, self.yij)

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = self.ue(self.mx, self.my)
        uet0 = sp.lambdify((x, y), ue.subs({t:t0}))(self.xij, self.yij)

        err = np.sqrt(self.h**2*np.sum((u[:] - uet0[:])**2))

        # print(f't: {t0:.3f}, error: {err:.3f}')
        return err

    def apply_bcs(self):
        self.Unp1[0, :] = 0
        self.Unp1[-1, :] = 0
        self.Unp1[:, 0] = 0
        self.Unp1[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.Nt = Nt
        self.c = c
        self.cfl =cfl
        self.mx = mx
        self.my = my
        self.create_mesh(N)
        dt = self.dt
        D2 = self.D2(N)/self.h**2
        self.initialize(N, mx, my)

        self.Unp1 = np.zeros((N+1, N+1))
        err = []


        self.plotdata = {0: self.Unm1.copy()}
        self.plotdata[1]= self.Un.copy()
        for n in range(1, Nt+1):
            self.ti = n*dt
            self.Unp1[:,:] = 2*self.Un - self.Unm1 + (c*dt)**2 * (D2 @ self.Un + self.Un @ D2.T)
            self.apply_bcs()

            self.Unm1[:] = self.Un
            self.Un[:] = self.Unp1

            err.append(self.l2_error(self.Unm1[:], self.ti))
            if n % store_data == 0:
                self.plotdata[n]= self.Unm1.copy()

        if store_data <= 0:
            return self.h, err
        else:
            return self.plotdata
        
    def animate(self):
        frames = []

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        for t, val in self.plotdata.items():
            # frame = ax.plot_wireframe(self.xij, self.yij, val)
            frame = ax.plot_surface(self.xij, self.yij, val, cmap=cm.coolwarm)

            frames.append([frame])
        ani = animation.ArtistAnimation(fig, frames, interval=150, blit=True,)
        ani.save("wave2d.gif", writer='pillow')
        plt.show()

    def convergence_rates(self, m=5, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2

        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Return second order differentiation matrix"""
        D2 = super().D2(N)
        D2[0, :2] = -2, 2
        D2[-1, -2:] = 2, -2
        return D2

    def ue(self, mx, my):
        """ Return the exact standing wave. """
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print(r)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    print(r)
    print(E)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol_Dir = Wave2D()
    sol_Neu = Wave2D_Neumann()
    _, E_Dir, _ = sol_Dir.convergence_rates(cfl=1/np.sqrt(2), mx=3, my=3)
    _, E_Neu, _ = sol_Neu.convergence_rates(cfl=1/np.sqrt(2), mx=3, my=3)
    assert np.all(E_Dir < 1e-12), "l2-feilen for Dirichet er ikke tilstrekkelig liten!"
    assert np.all(E_Neu < 1e-12), f"l2-feilen for Neumann er ikke tilstrekkelig liten!"

if __name__ == '__main__':
    wave = Wave2D()
    wave = Wave2D_Neumann()
    data = wave(80, 200, store_data=5)
    wave.animate()
    # test_convergence_wave2d()
    # test_convergence_wave2d_neumann()
    # test_exact_wave2d()
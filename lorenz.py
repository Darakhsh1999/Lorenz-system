import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class LorenzSystem():

    def __init__(
        self,
        n_paths: int = 1,
        sigma: float = 10,
        rho: float = 28,
        beta: float = 8/3,
        t_max: float = 50,
        dt: float = 0.01,
        L: float = 30):

        assert all([x > 0 for x in [sigma,rho,beta]]), "All parameters must be positive"

        self.n_paths = n_paths
        self.n_points = None
        self.N_quiver = 5 #
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.t_max = t_max
        self.dt = dt
        self.paths: list[dict] = []
        self.L = L
        q = np.sqrt(beta*(rho-1))
        self.FP = np.array([[0,0,0],[q,q,rho-1],[-q,-q,rho-1]])
        self.FP_COM = self.FP.mean(axis=0)
        self.limits =  np.vstack((self.FP_COM.T-L,self.FP_COM.T+L))

    def lorentz(t, X, sigma, rho, beta):
        """ Function handle for ODE solver """

        x,y,z = X

        x_dot = sigma*(y-x)
        y_dot = -x*z + rho*x - y
        z_dot = x*y - beta*z
        return (x_dot, y_dot, z_dot)
    
    def stability_matrix(self, X):

        x,y,z = X
        J = np.array([[-self.sigma, self.sigma,0],[-self.rho-z,-1,-x],[y,x,-self.beta]])
        return J
    
    def stability_eigenvalues(self, X):

        D = 4*self.rho*self.sigma + self.sigma**2 - 2*self.sigma + 1
        L_plus = 0.5*(-1-self.sigma+np.sqrt(D))
        L_minus = 0.5*(-1-self.sigma-np.sqrt(D))
        return np.array([-self.beta,L_plus,L_minus])
    
    def characteristic_equation(self, X):
        
        J = self.stability_matrix(X)
        J2 = np.matmul(J,J)

        det = np.linalg.det(J)
        tr = np.trace(J)
        tr2 = np.trace(J2)

        return (det,)
    
    

    def simulate_series(self, X0=None):
        """ Simulates a Lorenz dynamical system from starting point X0 """

        for i in range(self.n_paths):

            if X0 is None: # Random start point
                x0 = self.FP.mean(axis= 0) + 15*np.random.rand(3) # Around fix point C.O.M
            elif X0.shape[1] == self.n_paths : 
                x0 = X0[:,i]
            else:
                x0 = X0 + np.random.rand(3)

            lorentz_series = solve_ivp(
                fun=LorenzSystem.lorentz, 
                t_span=(0,self.t_max),
                y0=x0,
                args=(self.sigma, self.rho, self.beta),
                dense_output=True,
                t_eval=np.arange(0,self.t_max+self.dt,self.dt),
                max_step=self.dt)

            if self.n_points is None: self.n_points = lorentz_series.y.shape[1]
            self.paths.append({"pos": lorentz_series.y, "t": lorentz_series.t, "x0": x0})
    
    def figure(self, ax, label_level=2, q=100, frame_idx=None, taper=False):

        if frame_idx is not None: ax.set_title(f"T = {self.t_max*(frame_idx/self.n_points):.2f}")

        # Path LL0
        for path_idx, X in enumerate(self.paths):
            
            if frame_idx is None: # plot all points
                end_idx = slice(self.n_points)
            elif taper:
                end_idx = slice(max(frame_idx-q,0), frame_idx)
            else: # plot q last 
                end_idx = slice(frame_idx)

            if path_idx == 0:
                ax.plot3D(*X["pos"][:,end_idx], alpha= 0.6, label= "trajectory")
            else:
                ax.plot3D(*X["pos"][:,end_idx], alpha= 0.6)

        # Start and end points LL2
        if label_level >= 2:
            for path_idx, X in enumerate(self.paths):
                
                end_idx = -1 if frame_idx is None else frame_idx-1

                if path_idx == 0:
                    ax.scatter(*X["x0"], marker= "x", color= "blue", label= "start")
                    ax.scatter(*self.paths[path_idx]["pos"][:,end_idx], marker= "o", color= "blue", label= "end")
                else:
                    ax.scatter(*X["x0"], marker= "x", color= "blue")
                    ax.scatter(*self.paths[path_idx]["pos"][:,end_idx], marker= "o", color= "blue")


        # Fixed points LL3
        if label_level >= 3:
            ax.scatter(*self.FP.T, marker= "x", color= "black", label= "FP")

        # Vector field LL4
        if label_level >= 4:

            m_x, m_y, m_z = self.FP.mean(axis= 0) 
            x_mesh, y_mesh, z_mesh = np.meshgrid(
                np.linspace(m_x-self.L, m_x+self.L, self.N_quiver),
                np.linspace(m_y-self.L, m_y+self.L, self.N_quiver),
                np.linspace(m_z-self.L, m_z+self.L, self.N_quiver))

            x_dot = self.sigma*(y_mesh-x_mesh)
            y_dot = -x_mesh*z_mesh + self.rho*x_mesh - y_mesh
            z_dot = x_mesh*y_mesh - self.beta*z_mesh
            ax.quiver(x_mesh,y_mesh,z_mesh,x_dot,y_dot,z_dot, length= self.L/5, normalize= True)

        # Legend LL1
        if label_level >= 1: 
            ax.legend()

        ax.set_xlabel("x", fontsize= 15), ax.set_ylabel("y", fontsize= 15), ax.set_zlabel("z", fontsize= 15)
        ax.set_xlim(self.limits[:,0]), ax.set_ylim(self.limits[:,1]), ax.set_zlim(self.limits[:,2])

        return ax


    def show(self, label_level=1):
        """ Display image of trajectories """

        fig, ax = plt.subplots(subplot_kw= {'projection':'3d'})
        ax = self.figure(ax=ax, label_level=label_level)
        plt.show()

    def animate(self, label_level=1, taper=False, frame_skips= 5):
        """ Creates an animation from start to finish """
        
        pause_time = 0.01
        fig, ax = plt.subplots(subplot_kw= {'projection':'3d'})

        for frame_idx in range(0, self.n_points, frame_skips):
            t =+ pause_time
            ax.clear()
            ax = self.figure(ax=ax, label_level=label_level, frame_idx=frame_idx, taper=taper)
            plt.pause(pause_time)
        plt.show()

    

if __name__ == "__main__":

    lorenz = LorenzSystem(n_paths= 2, rho= 0.5)
    lorenz.simulate_series()
    #lorenz.show(label_level= 2)
    lorenz.animate(label_level= 2, taper=False)

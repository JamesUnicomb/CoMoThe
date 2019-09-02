import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from CollectiveMotion import CollectiveMotion


def data_for_cylinder_along_z(radius = 1.0, height = 0.5):
    z = np.linspace(-height/2, height/2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid)
    y_grid = radius*np.sin(theta_grid)
    return x_grid, y_grid, z_grid


def quiverplot_final_state(x,v,J):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    strtitle = 'J = %.03f' % (J)
    plt.suptitle(strtitle)
    
    
    sx, sy, sz = data_for_cylinder_along_z()
    s = ax.plot_surface(sx, sy, sz,  rstride=1, cstride=1, color='C0', alpha=0.2, linewidth=0)

    i = -1
    q = ax.quiver(x[i,:,0], x[i,:,1], x[i,:,2],
                  v[i,:,0], v[i,:,1], v[i,:,2])
                  
    ax.view_init(0.0, 0.0)

    plt.savefig(('figures/CollectiveMotion_'+str(J)+'.png'), dpi=80)

    plt.show()


def quiverplot_animation(x,v,J,lim=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    strtitle = 'J = %.03f' % (J)
    plt.suptitle(strtitle)

    ax.set_axis_off()

    i = 0
    q = ax.quiver(x[i,:,0], x[i,:,1], x[i,:,2],
                  v[i,:,0], v[i,:,1], v[i,:,2])
                  
    sx, sy, sz = data_for_cylinder_along_z()
    s = ax.plot_surface(sx, sy, sz,  rstride=1, cstride=1, color='C0', alpha=0.2, linewidth=0)

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    ax.view_init(0.0, 0.0)

    def update(i):
        print 'frame %d' % (i)
        
        ax.cla()

        q = ax.quiver(x[i,:,0], x[i,:,1], x[i,:,2],
                      v[i,:,0], v[i,:,1], v[i,:,2])
        s = ax.plot_surface(sx, sy, sz,  rstride=1, cstride=1, color='C0', alpha=0.2, linewidth=0)

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])

        ax.set_axis_off()

        return q,

    anim = FuncAnimation(fig, update, len(x), blit=False, interval=30)
    anim.save(('figures/CollectiveMotion_'+str(J)+'.mp4'), dpi=80, writer='imagemagick')
    plt.show()


def phase_change_plot(cm):
    va = []
    vm = []

    J_  = np.arange(0.0, 0.2, 0.03)
    J__ = np.repeat(J_, 1)
    
    for J in J__:
        va_ = cm.calculate_mean_velocity(J = J,
                                         N = 512,
                                         v0 = 0.05,
                                         n_steps = 400)

        va.append(va_)
        print J, va_

    va = np.array(va)
 
    for J in J_:
        vm.append(np.mean(va[J__ == J]))

    plt.scatter(J__, va, alpha=0.1)
    plt.plot(J_, vm)
    plt.savefig('figures/phase_change_J.png')
    plt.show()


def main():
    cm = CollectiveMotion()
    for J in (0.1,):
        t0 = time.time()
        x,v = cm.simulate_particles(J = J,
                                    N = 1024,
                                    n_steps = 600)
        t1 = time.time()
        print 'simulation took %.05f seconds' % (t1 - t0)
        quiverplot_final_state(x,v,J)
        quiverplot_animation(x,v,J)

    #phase_change_plot(cm)


if __name__=='__main__':
    main()

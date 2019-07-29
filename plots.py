import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from CollectiveMotion import CollectiveMotion


def quiverplot_final_state(x,v,J):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    strtitle = 'J = %.03f' % (J)
    plt.suptitle(strtitle)

    i = -1
    q = ax.quiver(x[i,:,0], x[i,:,1], x[i,:,2],
                  v[i,:,0], v[i,:,1], v[i,:,2])

    plt.savefig(('figures/CollectiveMotion_'+str(J)+'.png'), dpi=80)

    plt.show()


def quiverplot_animation(x,v,J,lim=0.5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    strtitle = 'J = %.03f' % (J)
    plt.suptitle(strtitle)

    ax.set_axis_off()

    i = 0
    q = ax.quiver(x[i,:,0], x[i,:,1], x[i,:,2],
                  v[i,:,0], v[i,:,1], v[i,:,2])


    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    def update(i):
        mx = np.mean(x[i,:,0])
        my = np.mean(x[i,:,1])
        mz = np.mean(x[i,:,2])

        print 'frame %d' % (i)
        
        ax.cla()

        q = ax.quiver(x[i,:,0], x[i,:,1], x[i,:,2],
                      v[i,:,0], v[i,:,1], v[i,:,2])

        ax.set_xlim([-lim + mx, lim + mx])
        ax.set_ylim([-lim + my, lim + my])
        ax.set_zlim([-lim + mz, lim + mz])

        ax.set_axis_off()

        return q,

    anim = FuncAnimation(fig, update, len(x), blit=False, interval=30)
    anim.save(('figures/CollectiveMotion_'+str(J)+'.gif'), dpi=80, writer='imagemagick')
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
    for J in (0.001, 0.07, 0.2):
        t0 = time.time()
        x,v = cm.simulate_particles(J = J,
                                    N = 512,
                                    n_steps = 400)
        t1 = time.time()
        print 'simulation took %.05f seconds' % (t1 - t0)
        quiverplot_final_state(x,v,J)
        #quiverplot_animation(x,v,J)

    #phase_change_plot(cm)


if __name__=='__main__':
    main()

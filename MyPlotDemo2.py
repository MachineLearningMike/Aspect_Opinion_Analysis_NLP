
import numpy as np

def distVec(A, B):
    C = np.power(A - B, 2) / 2
    return np.sum(C, axis = - 1)

def factsElems(facts):
     return facts[..., :-1], facts[..., -1]

def smElems(sm):
    return sm[..., :-1], sm[..., -1:]

def smLinPredict(sm, features):
  smW, smB = smElems(sm)
  return np.matmul( smW, np.transpose(features) ) + smB

def sigmoid(x, a):
    return 1 / (1 + np.exp(-a * x))
 
def loss(sm, facts):
    features, omLabels = factsElems(facts)
    smLabels = smLinPredict(sm, features)
    #smLabels = sigmoid( smLinPredict(sm, features), 0.1 )
    return distVec(smLabels, omLabels)


fact1 = np.array([1,5]); facts = np.array([ fact1 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

fact2 = np.array([3,25]); facts = np.array([ fact2 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

facts = np.array([ fact1, fact2 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

fact3 = np.array([5, 40]); fact4 = np.array([7, 50])
facts = np.array([ fact3, fact4 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

fact3 = np.array([5, 40]); fact4 = np.array([7, 50])
facts = np.array([fact1, fact2, fact3, fact4 ])
sm1 = np.array([-10, 25]); sm = np.array([sm1])
print("loss = {}".format(loss(sm, facts)))

facts = np.array([ fact1, fact2, fact3, fact4])
sm11 = np.array([-10, 25]); sm12 = np.array([-9, 20]); sm21 = np.array([-12, 20]); sm22 = np.array([-8, 28])
sm = np.array([ [sm11, sm12], [sm21, sm22] ])
print("loss = {}".format(loss(sm, facts)))


w = np.arange(3, 13, 1)
b = np.arange(-8, 18, 1)
W, B = np.meshgrid(w, b)
smW = W.reshape(tuple(np.append(W.shape,1)))
smB = B.reshape(tuple(np.append(B.shape,1)))
sm = np.append(smW, smB, axis = -1)

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

fact1 = np.array([1, 5]); fact2 = np.array([3, 25])
Z = loss(sm, np.array([fact1, fact2]))
fact3 = np.array([5, 40]); fact4 = np.array([7, 50])
Z1 = loss(sm, np.array([fact3, fact4]))
Z2 = loss(sm, np.array([fact1, fact2, fact3, fact4]))

thick = 5
width = (W.max()-W.min())/100*thick
depth = (B.max()-B.min())/100*thick
top = ((Z+Z1).max()-(Z+Z1).min())/100*thick


fig = plt.figure(dpi = 120)
ax = fig.gca(projection = '3d')

title = 'Loss by (w, b) for (fact1, fact2)'; zmax = Z2.max()
ax.set_title(title, fontsize=12, fontweight='normal', color='b')
ax.set_xlabel('w', fontsize=12, fontweight='normal', color='b')
ax.set_ylabel('b', fontsize=12, fontweight='normal', color='b')
#ax.set_zlabel('loss', fontsize=14, fontweight='normal', color='b')
ax.set_zlim(0, zmax)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))

norm = plt.Normalize(Z.min(), Z.max())
colors = cm.viridis(norm(Z))
rcount, ccount, _ = colors.shape
surf = ax.plot_surface(W, B, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))
ax.bar3d([10], [-5], [0], width, depth, top, color = 'magenta', shade = True)
ax.text(10+1, -5+1, 0, '0', color='magenta')

plt.show()
"""

def formatax(ax, title, zmax):
    ax.set_title(title, fontsize=12, fontweight='normal', color='b')
    ax.set_xlabel('w', fontsize=12, fontweight='normal', color='b')
    ax.set_ylabel('b', fontsize=12, fontweight='normal', color='b')
    #ax.set_zlabel('loss', fontsize=14, fontweight='normal', color='b')
    ax.set_zlim(0, zmax)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))
    #frame = ax.view_init(elev = ax.elev, azim = ax.azim - 60)

fig = plt.figure(dpi=120, figsize=plt.figaspect(0.33))
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax1 = fig.add_subplot(1, 3, 2, projection='3d')
ax2 = fig.add_subplot(1, 3, 3, projection='3d')

def init():   
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    title = 'Loss by (w, b) for (fact1, fact2)'; zmax = Z2.max()
    formatax(ax, title, zmax)
    surf = ax.plot_surface(W, B, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax.bar3d([10], [-5], [0], width, depth, top, color = 'magenta', shade = True)
    ax.text(10+1, -5+1, 0, '0', color='magenta')
    
    norm1 = plt.Normalize(Z1.min(), Z1.max())
    colors1 = cm.viridis(norm1(Z1))
    rcount1, ccount1, _ = colors1.shape
    title = 'Loss by (w, b) for (fact3, fact4)'; zmax = Z2.max()
    formatax(ax1, title, zmax)
    surf1 = ax1.plot_surface(W, B, Z1, rcount=rcount1, ccount=ccount1, facecolors=colors1, shade=False)
    surf1.set_facecolor((0,0,0,0))
    ax1.bar3d([5], [15], [0], width, depth, top, color = 'magenta', shade = True)
    ax1.text(5+1, 15+1, 0, '0', color='magenta')
    
    norm2 = plt.Normalize(Z2.min(), Z2.max())
    colors2 = cm.viridis(norm2(Z2))
    rcount2, ccount2, _ = colors2.shape
    title = 'Loss by (w, b) for (fact1, ..., fact4)'; zmax = Z2.max()
    formatax(ax2, title, zmax)
    surf2 = ax2.plot_surface(W, B, Z2, rcount=rcount2, ccount=ccount2, facecolors=colors2, shade=False)
    surf2.set_facecolor((0,0,0,.1))
    ax2.bar3d([6], [5], [15], width, depth, top, color = 'magenta', shade = True)
    ax2.text(6+1, 5+1, 0, '15', color='magenta')
    
    print('init elev = {}, azim = {}'.format(ax.elev, ax.azim))

    return fig

#init()
#plt.show()


init_elev = 30; init_azim = -120
slow = 2
maxAngle = 90 # degree
rest = 0
def animate(i):
    x = 1.0 * i / slow
    if x <= maxAngle:
        e = maxAngle * ( 1.0 - np.cos(np.pi * x / maxAngle))
    elif x < maxAngle + rest:
        e = maxAngle
    elif x < 2 * maxAngle + rest:
        e = maxAngle * ( 1.0 + np.cos(np.pi * (x - maxAngle - rest) / maxAngle )) #2 * maxAngle + rest - i
    else:
        e = 0    
    
    a = e
    ax.view_init(elev = init_elev + e, azim = init_azim + a)
    ax1.view_init(elev = init_elev + e, azim = init_azim + a)
    ax2.view_init(elev = init_elev + e, azim = init_azim + a)
    return fig

init()


# Animate
ani = animation.FuncAnimation(fig, animate, init_func=init, frames= slow * 2 * (maxAngle + rest), interval=100, blit=False)
FFwriter = animation.FFMpegWriter(fps=25, extra_args=['-vcodec', 'libx264']); 
ani.save('animation.mp4', writer = FFwriter)
#ani.save("animation.gif", writer="imagemagick")

plt.show()
"""

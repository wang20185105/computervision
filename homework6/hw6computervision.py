from camera import Camera as camera
from PIL import Image
from pylab import *
from numpy import *
from mpl_toolkits.mplot3d import axes3d

# load some images
im1 = array(Image.open('D:/computervision/images/001.jpg'))
im2 = array(Image.open('D:/computervision/images/002.jpg'))
# load 2D points for each view to a list
points2D = [loadtxt('D:/computervision/2D/00'+str(i+1)+'.corners').T for i in range(3)]
# load 3D points
points3D = loadtxt('D:/computervision/3D/p3d').T
# load correspondences
corr = genfromtxt('D:/computervision/2D/nview-corners',dtype='int',missing_values='*')
# load cameras to a list of Camera objects
#print(loadtxt('D:/computervision/2D/001.P'))
#P = camera(loadtxt('D:/computervision/2D/001.P'))
P = [camera(loadtxt('D:/computervision/2D/00'+str(i+1)+'.P')) for i in range(3)]

# make 3D points homogeneous and project
X = vstack( (points3D,ones(points3D.shape[1])) )
x = P[0].project(X)
# plotting the points in view 1
figure()
imshow(im1)
plot(points2D[0][0],points2D[0][1],'*')
axis('off')
figure()
imshow(im1)
plot(x[0],x[1],'r.')
axis('off')
show()

fig = figure()
ax = fig.gca(projection="3d")
# generate 3D sample data
X,Y,Z = axes3d.get_test_data(0.25)
# plot the points in 3D
ax.plot(X.flatten(),Y.flatten(),Z.flatten(),'o')
show()

fig = figure()
ax = fig.gca(projection='3d')
ax.plot(points3D[0],points3D[1],points3D[2],'k.')
show()

def compute_fundamental(x1,x2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm.
    each row is constructed as
    [x’*x, x’*y, x’, y’*x, y’*y, y’, x, y, 1] """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don’t match.")
    # build matrix for equations
    A = zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
            x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
            x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U,dot(diag(S),V))
    return F

def compute_epipole(F):
    """ Computes the (right) epipole from a
    fundamental matrix F.
    (Use with F.T for left epipole.) """
    # return null space of F (Fx=0)
    U,S,V = linalg.svd(F)
    e = V[-1]
    return e/e[2]


def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
    in an image. F is the fundamental matrix
    and x a point in the other image."""
    m,n = im.shape[:2]
    line = dot(F,x)
    # epipolar line parameter and values
    t = linspace(0,n,100)
    lt = array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
    # take only line points inside the image
    ndx = (lt>=0) & (lt<m)
    plot(t[ndx],lt[ndx],linewidth=2)
    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')

# index for points in first two views
ndx = (corr[:,0]>=0) & (corr[:,1]>=0)
# get coordinates and make homogeneous
x1 = points2D[0][:,corr[ndx,0]]
x1 = vstack( (x1,ones(x1.shape[1])) )
x2 = points2D[1][:,corr[ndx,1]]
x2 = vstack( (x2,ones(x2.shape[1])) )
# compute F
F = compute_fundamental(x1,x2)
# compute the epipole
e = compute_epipole(F)
# plotting
figure()

imshow(im1)
# plot each line individually, this gives nice colors
for i in range(5):
    plot_epipolar_line(im1,F,x2[:,i],e,False)
axis('off')
figure()
imshow(im2)
# plot each point individually, this gives same colors as the lines
for i in range(5):
    plot(x2[0,i],x2[1,i],'o')
axis('off')
show()


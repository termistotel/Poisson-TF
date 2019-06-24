import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import json

# Width and height of grid
x, y = 100, 100

def sigmoid(x, k=1):
	return 1/(np.exp(x/k) +1)

def softDisc(softness, r0, r):
	k = softness

	# Approximately homogenously charged disc with radious r0
	# Edges are softened by sigmoid
	rho = sigmoid(r-r0, softness+1e-7)	

	return rho

def gaussNoise(x, y, mu = 0, sigma = 1):
	return (np.random.randn(x, y) + mu)*sigma

def randomNoise(x, y, minimum = 0, maximum = 1):
	return np.random.rand(x, y)*(maximum-minimum) + minimum

# Function to fix images for cv2 video save
def fixImage(img, rel = True):
	if rel:
		mn = np.min(img)
		mx = np.max(img)
		return (255/(mx-mn)*(img-mn)*np.ones((1,1,3))).astype(np.uint8) 
	else:
		return (255*np.clip(img, 0, 1)*np.ones((1,1,3))).astype(np.uint8)

def gauss(xx,mu,sigma):
    a = np.exp(-np.square((xx-mu)/sigma)/2)
    N = np.sqrt(2*np.pi)*sigma
    return a/N

# Hparameters
niter = 20000				# Number of iterations
r0 = 5						# Disc radious
softness = 0.1				# Disc softness
gamma = 1
D = 0.1						# Difussion coefficient
seed = 1337

# Hparameters for steprate control
dt0 = 1*1e-1					# Initial step rate
rate = 1.0
tau = 10000

# Hparams for square stuff
alfa = 1e-2
betaCurrent = 1
betaSquare = 1

# MetaData
meta = {"niter": niter, "seed": seed, "gamma": gamma, "D": D, "dt0": dt0, "alfa": alfa}

# Numpy random seed
np.random.seed(seed)

# Cartesian mesh coordinates
xx, yy = np.meshgrid(range(x), range(y))

# Polar mesh radius
r = np.sqrt(np.square(xx-x//2) + np.square(yy-y//2))

# Generate starting distribution
# c0 = softDisc(softness, r0, r)
# c0 = gaussNoise(x, y)
c0 = randomNoise(x, y, minimum=-1, maximum=1)

step = ()

# Hough init
rmax = int(np.sqrt(np.sum(np.square([x, y]))))
thetaNum = 40
rNum = 40
sigma = 1

theta = np.linspace(-np.pi, np.pi, thetaNum).reshape(1, 1,1,-1)
radii = np.linspace(0, rmax, rNum).reshape(1, 1,-1,1)

xx, yy = np.meshgrid(np.arange(x), np.arange(y))
xx = xx.reshape(xx.shape + (1,1))
yy = yy.reshape(yy.shape + (1,1))

# xx1 = (xx*np.cos(theta) + yy*np.sin(theta)).astype(np.int)
xx1 = xx*np.cos(theta) + yy*np.sin(theta)

# Define tensorflow graph
graph = tf.Graph()
with graph.as_default():

	# Main variable, distribution at starting time in tensorflow
	c = tf.Variable(c0.reshape(1,x,y,1), dtype=tf.float32)

	# Total ammount
	C = tf.reduce_mean(c)

	# Getting laplacian of every point in c
	nabla2 = tf.constant(np.array([[1,1,1],[1,-8,1],[1,1,1]]).reshape(3,3,1,1), dtype=tf.float32)
	nabla2c = tf.nn.convolution(c, nabla2, "SAME")

	# Chemical potential mu = c^3 - c - gamma* nabla^2 (c)
	mu = tf.pow(c, 3) - c - gamma*nabla2c

	# dy = tf.constant(0.5*np.array([[0,-1,0],[0,0,0],[0,1,0]]).reshape(3,3,1,1), dtype=tf.float32)
	# dx = tf.constant(0.5*np.array([[0,0,0],[-1,0,1],[0,0,0]]).reshape(3,3,1,1), dtype=tf.float32)
	# currentx = tf.nn.convolution(mu, dx, "SAME")
	# currenty = tf.nn.convolution(mu, dy, "SAME")
	# currentMag = tf.sqrt(tf.square(currentx) + tf.square(currenty))

	# Sobel
	dy = tf.constant(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).reshape(3,3,1,1), dtype=tf.float32)
	dx = tf.constant(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).reshape(3,3,1,1), dtype=tf.float32)
	edgex = tf.nn.convolution(c, dx, "SAME")
	edgey = tf.nn.convolution(c, dy, "SAME")
	edge = tf.sqrt(tf.square(edgex) + tf.square(edgey))
	maps = tf.constant( gauss(xx1, radii, sigma), dtype=tf.float32)

	hough = tf.sum(edge*maps, axis=[0,1])

	# Getting laplacian of every point in mu
	nabla2mu = tf.nn.convolution(mu, nabla2, "SAME")

	# Time derivative
	dcdt = D*nabla2mu

	# Time step
	dt = dt0

	# Iteration by discretization of HC equation: dc / dt = D * nabla^2 (mu)
	# Iteration step c[i+1] := c[i] + dt* D * nabla^2 (mu)
	step +=(c.assign(c + dt*dcdt - C),)

	# Trying to make it square-y

	# Making current larger
	# step2 = (tf.train.AdamOptimizer(learning_rate = alfa).minimize(loss),)

	# Initialization of variables
	var_init = tf.global_variables_initializer()


# Saving results as videos
num = len(os.listdir('videos'))//6

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(os.path.join('videos', str(num) + 'Concentration' + '.avi'),fourcc, 20.0, (x, y))
out2 = cv2.VideoWriter(os.path.join('videos', str(num) + 'Chempot' + '.avi'),fourcc, 20.0, (x, y))
out3 = cv2.VideoWriter(os.path.join('videos', str(num) + 'TimeDeriv' + '.avi'),fourcc, 20.0, (x, y))
out4 = cv2.VideoWriter(os.path.join('videos', str(num) + 'Concbin' + '.avi'),fourcc, 20.0, (x, y))
out5 = cv2.VideoWriter(os.path.join('videos', str(num) + 'Current' + '.avi'),fourcc, 20.0, (x, y))

# Start session and initialize variables
sess = tf.Session(graph = graph)
sess.run(var_init)

for i in range(niter):
	if i%100 == 0:

		# Write potential, electric field and charge density reconstruction to videos
		concet = sess.run(c)[0,:,:,:]
		conbin = (sess.run(c)[0,:,:,:]>0).astype(np.float)
		chemPot = sess.run(mu)[0,:,:,:]
		timeDer = sess.run(dcdt)[0,:,:,:]
		current = sess.run(currentMag)[0,:,:,:]

		out1.write(fixImage(concet, rel = True))
		out2.write(fixImage(chemPot, rel = True))
		out3.write(fixImage(timeDer, rel = True))
		out4.write(fixImage(conbin, rel=True))
		out5.write(fixImage(current, rel=True))

		# if i%1000 ==0:
		# 	plt.imshow(concet[:,:,0], cmap='gray')
		# 	plt.show()
		# 	plt.imshow(conbin[:,:,0], cmap='gray')
		# 	plt.show()
		# 	plt.imshow(chemPot[:,:,0], cmap='gray')
		# 	plt.show()
		# 	plt.imshow(timeDer[:,:,0], cmap='gray')
		# 	plt.show()

	sess.run(step + step2)
	print(i, sess.run([C, lossTotal]))

out1.release()
out2.release()
out3.release()
out4.release()
out5.release()

with open(os.path.join("videos", str(num) + 'hparameters'), 'w') as file:
	json.dump(meta, file)

sess.close()
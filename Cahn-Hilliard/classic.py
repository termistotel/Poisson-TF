import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

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

# Hparameters
niter = 50000				# Number of iterations
r0 = 5						# Disc radious
softness = 0.1				# Disc softness
gamma = 1
D = 0.1						# Difussion coefficient

# Hparameters for steprate control
dt0 = 1e-1					# Initial step rate
rate = 1.0
tau = 10000

# Cartesian mesh coordinates
xx, yy = np.meshgrid(range(x), range(y))

# Polar mesh radius
r = np.sqrt(np.square(xx-x//2) + np.square(yy-y//2))

# Generate starting distribution
# c0 = softDisc(softness, r0, r)
# c0 = gaussNoise(x, y)
c0 = randomNoise(x, y, minimum=-1, maximum=1)

# Display starting distribution
plt.imshow(c0, cmap='gray')
plt.show()

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

	# Getting laplacian of every point in mu
	nabla2mu = tf.nn.convolution(mu, nabla2, "SAME")

	# Time derivative
	dcdt = D*nabla2mu

	# Time step
	gs = tf.Variable(0, dtype=tf.float32)
	dt = dt0 * tf.pow(rate, gs/tau)

	# Iteration by discretization of HC equation: dc / dt = D * nabla^2 (mu)
	# Iteration step c[i+1] := c[i] + dt* D * nabla^2 (mu)
	step = (c.assign(c + dt*dcdt - C), gs.assign(gs+1))

	# Initialization of variables
	var_init = tf.global_variables_initializer()


# Saving results as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('concentration0.avi',fourcc, 20.0, (x, y))
out2 = cv2.VideoWriter('chempot0.avi',fourcc, 20.0, (x, y))
out3 = cv2.VideoWriter('timeDeriv0.avi',fourcc, 20.0, (x, y))
out4 = cv2.VideoWriter('concbin0.avi',fourcc, 20.0, (x, y))

# Start session and initialize variables
sess = tf.Session(graph = graph)
sess.run(var_init)

for i in range(niter):
	if i%10 == 0:

		# Write potential, electric field and charge density reconstruction to videos
		concet = sess.run(c)[0,:,:,:]
		conbin = (sess.run(c)[0,:,:,:]>0).astype(np.float)
		chemPot = sess.run(mu)[0,:,:,:]
		timeDer = sess.run(dcdt)[0,:,:,:]

		out1.write(fixImage(concet, rel = True))
		out2.write(fixImage(chemPot, rel = True))
		out3.write(fixImage(timeDer, rel = True))
		out4.write(fixImage(conbin, rel=True))

		# if i%1000 ==0:
		# 	plt.imshow(concet[:,:,0], cmap='gray')
		# 	plt.show()
		# 	plt.imshow(conbin[:,:,0], cmap='gray')
		# 	plt.show()
		# 	plt.imshow(chemPot[:,:,0], cmap='gray')
		# 	plt.show()
		# 	plt.imshow(timeDer[:,:,0], cmap='gray')
		# 	plt.show()

	sess.run(step)
	print(i, sess.run(C))

out1.release()
out2.release()
out3.release()

sess.close()
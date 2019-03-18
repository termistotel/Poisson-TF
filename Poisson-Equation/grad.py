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
u0 = np.zeros((1,x,y,1))	# Boundary conditions and 0th iteration solution
r0 = 5						# Charged disc radious
softness = 0.1				# Disc softness

beta = 1					# Beta
alpha = 0.002				# Initial learning rate
tau = 50000					# Number of steps until alpha drops by 1/base
base = 0.1					# Base of exponential decay of learning rate

# Cartesian mesh coordinates
xx, yy = np.meshgrid(range(x), range(y))

# Polar mesh radius
r = np.sqrt(np.square(xx-x//2) + np.square(yy-y//2))

# Generate charge density
pn = softDisc(softness, r0, r)

# Display charge density
plt.imshow(pn)
plt.show()

# Define tensorflow graph
graph = tf.Graph()
with graph.as_default():

	# Main variable
	d = tf.Variable(np.zeros((1,x,y,1)), dtype=tf.float32)

	# mapa will be used to sellect vrelevant variables
	mapa = tf.constant((r<min(x//2,y//2)).reshape(1,x,y,1),  dtype=tf.float32)

	# Boundary conditions
	ru = tf.constant(u0, dtype = tf.float32)

	# Only relevant variables are used
	u = ru + d*mapa

	# Charge density in tensorflow graph
	pt = tf.constant(pn.reshape(1,x,y,1), dtype=tf.float32)

	# Getting neighborhood of every point on the grid
	neigfil = tf.constant(np.array([[1,1,1],[1,0,1],[1,1,1]]).reshape(3,3,1,1), dtype=tf.float32)
	neighborhood = tf.nn.convolution(u, neigfil, "SAME")

	# Reconstructing charge density from curent potential
	recons = -(neighborhood - 8*u)

	# Aproximate distance from the goal
	loss1 = tf.reduce_mean(tf.square(pt - recons))
	loss15 = tf.reduce_max(tf.square(pt - recons))
	loss2 = tf.sqrt(tf.reduce_mean(tf.square(pt - recons)))
	loss25 = tf.sqrt(tf.reduce_max(tf.square(pt - recons)))
	loss3 = tf.reduce_mean(tf.abs(pt - recons))
	loss35 = tf.reduce_max(tf.abs(pt - recons))

	loss = beta*loss2 + (1-beta)*loss25

	# Variable learning rate
	gs = tf.Variable(0, trainable=False)
	lr = tf.train.exponential_decay(alpha, gs, tau, base)

	# Iteration by minimizing loss function
	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	step = optimizer.minimize(loss, global_step = gs)

	# Calculate electric field by approximating differentiation with sobel filter
	Ex = tf.nn.convolution(u, tf.constant(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).reshape(3,3,1,1), dtype=tf.float32), "SAME")
	Ey = tf.nn.convolution(u, tf.constant(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).reshape(3,3,1,1), dtype=tf.float32), "SAME")
	# Magnitude of electric field
	E = tf.sqrt(tf.square(Ex) + tf.square(Ey))

	# Initialization of variables
	var_init = tf.global_variables_initializer()



# Saving results as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('potential1.avi',fourcc, 20.0, (x, y))
out2 = cv2.VideoWriter('elfield1.avi',fourcc, 20.0, (x, y))
out3 = cv2.VideoWriter('recons1.avi',fourcc, 20.0, (x, y))

# Start session and initialize variables
sess = tf.Session(graph = graph)
sess.run(var_init)

for i in range(niter):
	if i%1000 == 0:

		# Write potential, electric field and charge density reconstruction to videos
		pot = fixImage(sess.run(u)[0,:,:,:])
		elf = fixImage(sess.run(E)[0,:,:,:])
		rec = sess.run(recons)[0,:,:,:]

		print(np.max(sess.run(u)))

		out1.write(pot)
		out2.write(elf)
		out3.write(fixImage(rec, rel = True))

		# if i%1000 ==0:
		# 	plt.imshow(pot, cmap='gray')
		# 	plt.show()
		# 	plt.imshow(elf, cmap='gray')
		# 	plt.show()
		# 	plt.imshow(pn, cmap='gray')
		# 	plt.show()
		# 	plt.imshow(rec[:,:,0], cmap='gray')
		# 	plt.show()

	sess.run(step)
	print(i, sess.run([step, loss]))

out1.release()
out2.release()
out3.release()

sess.close()
from mpi4py import MPI

comm = MPI.COMM_WORLD

nproc = comm.Get_size()
# The first processor is leader, so one fewer available to be a worker
nworkers = nproc - 1

# samples
N = 100000000
delta = 1.0 / N

# integral
I = 0.0

def integrand(x):
  return(4.0 / (1.0 + x*x))

if comm.Get_rank() == 0:

  # Leader: choose points to sample function, send to workers and
  # collect their contributions. Also calculate a sub-set of points.

  for i in range(0,N):

    # decide which rank evaluates this point
    j = i % nproc

    # mid-point rule
    x = (i+0.5) * delta

    if j == 0:
      # so do this locally using the leader machine
      y = integrand(x) * delta
    else:
      # communicate to a worker
      comm.send(x, dest=j)
      y = comm.recv(source=j)

    I += y

  # Shut down the workers
  for i in range(1, nproc):
    comm.send(-1.0, dest=i)

  print("Integral %.10f" % I)

else:

  # Worker: waiting for something to happen, then stop if sent message
  # outside the integral limits

  while True:

    x = comm.recv(source=0)

    if x < 0.0:
      # stop the worker
      break

    else:
      comm.send(integrand(x) * delta, dest=0)

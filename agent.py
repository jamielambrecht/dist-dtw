from mpi4py import MPI
import numpy as np
from numpy.lib.shape_base import column_stack

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

dimensions = np.array((0,0))
comm.Bcast(dimensions, root=0)

M, N = dimensions[0], dimensions[1]

complete = False
reqs = []

counter = 0

if rank == 0:
    limit = N
if rank == 1:
    limit = M

complete_bit = np.array(0, dtype=np.int64)
comm.Irecv(complete_bit, source=0, tag=6)

# need the length of the previous data before we can set up buffers
previous_data_size = np.array(0, dtype=np.int64)

while counter < limit - 1:
    comm.Recv(previous_data_size, source=0, tag=1)
    print(str(previous_data_size) + "!")
    # set up buffers
    previous_data = np.zeros(previous_data_size, dtype=np.int64)
    cost_array = np.zeros(previous_data_size - 1, dtype=np.int64)
    neighbor = np.array(0, dtype=np.int64)
    # receive essential data
    reqs.append(comm.Irecv(previous_data, source=0, tag=2))
    reqs.append(comm.Irecv(neighbor, source=0, tag=3))
    reqs.append(comm.Irecv(cost_array, source=0, tag=4))
    # synchronize
    for req in reqs:
        req.wait()

    # create a buffer for our calculation
    new_data = np.zeros(previous_data_size - 1, dtype=np.int64)
    for i in range(len(new_data)):
        new_data[i] = min(previous_data[i], neighbor, previous_data[i + 1]) + cost_array[i]
        neighbor = new_data[i]
    print(str(rank) + " returning: " + str(new_data))
    comm.Send(new_data, dest=0, tag=5)
    previous_data_size = np.array(0, dtype=np.int64)

    if (complete_bit > 0):
        break

comm.Disconnect()
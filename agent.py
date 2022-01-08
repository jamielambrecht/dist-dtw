from mpi4py import MPI
import numpy as np
from numpy.lib.shape_base import column_stack
import sys, time

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

complete_tog = np.array(0, dtype=np.int64)

# need the length of the previous data before we can set up buffers
previous_data_size = np.array(0, dtype=np.int64)

prev_data_req = comm.Irecv(previous_data_size, source=0, tag=1)
complete_flag_req = comm.Irecv(complete_tog, source=0, tag=7)

new_data_send = None

while complete_tog < 1:

    if new_data_send != None:
        if MPI.Request.Test(new_data_send) == True:
            prev_data_req = comm.Irecv(previous_data_size, source=0, tag=1)
            new_data_send = None

    if prev_data_req != None:
        if MPI.Request.Test(prev_data_req) == True:
            # set up buffers
            previous_data = np.zeros(previous_data_size, dtype=np.int64)
            cost_array = np.zeros(previous_data_size - 1, dtype=np.int64)
            neighbor = np.array(0, dtype=np.int64)
            # receive essential data
            reqs.append(comm.Irecv(previous_data, source=0, tag=2))
            reqs.append(comm.Irecv(neighbor, source=0, tag=3))
            reqs.append(comm.Irecv(cost_array, source=0, tag=4))
            # synchronize
            prev_data_req = None

    if MPI.Request.Testall(reqs) == True and len(reqs) > 0:
        # create a buffer for our calculation
        new_data = np.zeros(int(previous_data_size) - 1, dtype=np.int64)
        for i in range(len(new_data)):
            new_data[i] = min(previous_data[i], neighbor, previous_data[i + 1]) + cost_array[i]
            neighbor = new_data[i]
        print(str(rank) + " returning: " + str(new_data))
        new_data_send = comm.Isend(new_data, dest=0, tag=5 + rank)
        previous_data_size = np.array(0, dtype=np.int64)
        reqs.clear()

    if MPI.Request.Test(complete_flag_req) == True or complete_tog == 1:
        break

print(f"Agent {rank} exited.")
# comm.Disconnect()
sys.exit(0)
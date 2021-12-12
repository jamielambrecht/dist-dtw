from mpi4py import MPI
import numpy as np
import sys

def matrix(value, M, N):

    matrix = np.zeros((M, N), np.int64)
    return matrix

comm = MPI.COMM_SELF.Spawn(sys.executable, args=['agent.py'], maxprocs=2)
rank = comm.Get_rank()
size = comm.Get_size()

A = [(2,2), (0,4), (2,6), (4,5)]
B = [(1,1), (0,6), (4,4)]

M = len(A)
N = len(B)

m_n = np.array((M,N))
print("About to send matrix dimensions (M x N): " + str(m_n))
comm.Bcast(m_n, root=MPI.ROOT)

# Manager Process initializes matrices.
cost_matrix = matrix(0, M, N) # Initialize with zeros

for i in range(M):
    for j in range(N):
        cost_matrix[i,j] = (A[i][0] - B[j][0])**2 + (A[i][1] - B[j][1])**2 # distance function

print(cost_matrix)

dtw_matrix = matrix(0, M, N) # Initialize with zeros
dtw_matrix[0, 0] = cost_matrix[0, 0] # Initialize top left cell

neighbor = dtw_matrix[0, 0]
# COLUMN
for i in range(1, M):
    dtw_matrix[i, 0] = neighbor + cost_matrix[i, 0]
    neighbor = dtw_matrix[i, 0]
    print("M @ " + str(i) + ": " + str(dtw_matrix[i, 0]))
# ROW
neighbor = dtw_matrix[0, 0]
for i in range(1, N):
    dtw_matrix[0, i] = neighbor + cost_matrix[0, i]
    neighbor = dtw_matrix[0, i]
    print("N @ " + str(i) + ": " + str(dtw_matrix[i, 0]))

print(dtw_matrix)

send_row, send_column = True, True
req_for_row, req_for_col = [], []

row_counter, col_counter = 1, 1
row_limit, col_limit = N - 1, M - 1

complete = False
first_time = True

while row_counter < row_limit or col_counter < col_limit:
    
    if send_row:
        # Stuff that we need to send to the row process
        previous_row = np.array(dtw_matrix[row_counter - 1, col_counter - 1:], dtype=np.int64)
        cost_row = np.array(cost_matrix[row_counter, col_counter:], dtype=np.int64)
        left_neighbor = np.array(dtw_matrix[row_counter, col_counter - 1], dtype=np.int64)

        req_for_row.append(comm.Isend(np.array(len(previous_row), dtype=np.int64), dest=0, tag=1))
        req_for_row.append(comm.Isend(previous_row, dest=0, tag=2))
        req_for_row.append(comm.Isend(left_neighbor, dest=0, tag=3))

        new_row = np.zeros(len(previous_row) - 1, dtype = np.int64)
        req_for_row.append(comm.Isend(cost_row, dest=0, tag=4))

    if send_column:
        previous_col = np.array(dtw_matrix[row_counter - 1:, col_counter - 1], dtype=np.int64)
        cost_col = np.array(cost_matrix[row_counter:, col_counter], dtype=np.int64)
        up_neighbor = np.array(dtw_matrix[row_counter - 1, col_counter], dtype=np.int64)

        req_for_col.append(comm.Isend(np.array(len(previous_col), dtype=np.int64), dest=1, tag=1))
        req_for_col.append(comm.Isend(previous_col, dest=1, tag=2))
        req_for_col.append(comm.Isend(up_neighbor, dest=1, tag=3))

        new_col = np.zeros(len(previous_col) - 1, dtype = np.int64)
        req_for_col.append(comm.Isend(cost_col, dest=1, tag=4))

    if len(req_for_row) > 0:
        for req in req_for_row:
            req.wait()
            req_for_row.clear()
            req_for_row.append(comm.Irecv(new_row, source = 0, tag=5))
    if len(req_for_col) > 0:
        for req in req_for_col:
            req.wait()
            req_for_col.clear()
            req_for_col.append(comm.Irecv(new_col, source = 1, tag=5))

    if len(req_for_row) > 0:
        for req in req_for_row:
            req.wait()
            req_for_row.clear()
        for i in range(len(new_row)):
            dtw_matrix[row_counter, col_counter + i] = new_row[i]

    if len(req_for_col) > 0:
        for req in req_for_col:
            req.wait()
            req_for_col.clear()
        for i in range(len(new_col)):
            dtw_matrix[row_counter + i, col_counter] = new_col[i]


    if row_counter < row_limit:
        row_counter += 1
        if row_counter < row_limit:
            send_row = True
        else:
            comm.Send(np.array(1, dtype=np.int64), dest=0, tag=6)
            print("Sent sigterm to 0")

    if col_counter < col_limit:
        col_counter += 1
        if col_counter < col_limit:
            send_column = True
        else:
            comm.Send(np.array(1, dtype=np.int64), dest=1, tag=6)
            print("Sent sigterm to 1")

    print(dtw_matrix)

    print("Computed:: Rows: {}, Cols: {}".format(row_counter, col_counter))

print("RESULT:\n" + str(dtw_matrix))

comm.Disconnect()
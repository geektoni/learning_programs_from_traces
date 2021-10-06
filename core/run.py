from core.mcts_exact import MCTSExact
from core.policy import Policy
from core.environment import Environment

if __name__ == "__main__":

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        mcts = MCTSExact(Environment(), Policy(), 1)
    else:
        mcts = None

    mcts = comm.bcast(mcts, root=0)

    traces = mcts.sample_execution_trace()

    traces = comm.gather(traces, root=0)

    if rank == 0:
        traces = [t.get_id() for t in traces]
        print(traces)


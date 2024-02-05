import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def dfs(mst, visited, vertex):
    """
    Depth First Search to mark all vertices reachable from a given starting vertex.

    Args:
    - mst: The MST adjacency matrix
    - visited: List to keep track of visited vertices
    - vertex: The current vertex to explore
    """
    visited[vertex] = True
    for i, edge in enumerate(mst[vertex]):
        if edge > 0 and not visited[i]:
            dfs(mst, visited, i)

def check_connectedness(mst, num_vertices):
    """
    Checks if the MST is connected by ensuring there's a path from any vertex to every other vertex.

    Args:
    - mst: The MST adjacency matrix
    - num_vertices: Number of vertices in the graph
    """
    visited = [False] * num_vertices
    dfs(mst, visited, 0)  # Start DFS from vertex 0
    assert all(visited), 'MST is not connected'

def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # Check for total weight
    total_weight = np.sum(mst) / 2  # Divide by 2 because it's an undirected graph and edges are counted twice
    assert approx_equal(total_weight, expected_weight), f'Proposed MST weight differs from expected by more than {allowed_error}'

    # Check the number of edges (N-1 for N vertices)
    num_edges = np.count_nonzero(mst) / 2  # Divide by 2 to adjust for symmetric matrix counting edges twice
    num_vertices = adj_mat.shape[0]
    assert num_edges == num_vertices - 1, 'Proposed MST does not have N-1 edges'

    # Check for connectedness
    check_connectedness(mst, num_vertices)

    # Ensure all edges in the MST exist in the original graph and their weights are correct
    # for i in range(mst.shape[0]):
    #     for j in range(i + 1, mst.shape[1]):
    #         if mst[i, j] > 0:  # If there's an edge in the MST
    #             assert adj_mat[i, j] == mst[i, j], 'Edge weight in MST does not match the original graph'

    # Ensure all edges in the MST exist in the original graph and their weights are correct
    # for i in range(mst.shape[0]):
    #     for j in range(i + 1, mst.shape[1]):
    #         if mst[i, j] > 0:  # If there's an edge in the MST
    #             assert adj_mat[i, j] == mst[i, j], 'Edge weight in MST does not match the original graph'
    

def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    # Create a simple graph with 4 vertices
    adj_mat = np.array([
        [0, 1, 4, 0],
        [1, 0, 2, 3],
        [4, 2, 0, 5],
        [0, 3, 5, 0]
    ])
    expected_weight = 6  # Known weight of MST for this graph
    g = Graph(adj_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, expected_weight)

    pass

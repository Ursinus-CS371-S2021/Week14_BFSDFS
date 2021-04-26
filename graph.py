import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import deque
from heap import *

class GraphNode:
    def __init__(self, idx=-1):
        self.idx = idx
        self.edges = [] # List of nodes that are adjacent
        self.dists = [] # Parallel list of distances of the edges to these nodes
        self.data = {}
        self.visited = False
        self.touched = False
    
    def __str__(self):
        return "{}".format(self.idx)

def draw_2d_graph(nodes, edges, draw_nodes=True, draw_labels=True, linewidth=1):
    ax = plt.gca()
    ax.set_facecolor((0.9, 0.9, 0.9))
    for (i, j, d) in edges:
        x1, y1 = nodes[i].data['x'], nodes[i].data['y']
        x2, y2 = nodes[j].data['x'], nodes[j].data['y']
        plt.plot([x1, x2], [y1, y2], linewidth=linewidth, c='k')
    for i, n in enumerate(nodes):
        if draw_nodes:
            sz = 50
            if n.visited:
                sz = 100
            c = 'g'
            if n.touched:
                if n.visited:
                    c = 'k'
                else:
                    c = 'r'
            plt.scatter(n.data['x'], n.data['y'], sz, c=c)
            if draw_labels:
                plt.text(n.data['x']+0.002, n.data['y']+0.002, "{}".format(i), zorder=10, c=c, fontsize='large')
                
def draw_grid_graph(nodes, edges, draw_nodes=True, draw_labels=True, draw_diags=True):
    N = int(np.sqrt(len(nodes)))
    ax = plt.gca()
    ax.set_facecolor((0.9, 0.9, 0.9))
    # Draw horizontal edges
    for i in range(N):
        plt.plot([i, i], [0, N-1], 'k')
        plt.plot([0, N-1], [i, i], 'k')
    # Draw diagonal edges
    if draw_diags:
        for i in range(N):
            plt.plot([0, i], [i, 0], 'k')
            plt.plot([N-1, i], [i, N-1], 'k')
            plt.plot([0, N-i-1], [i, N-1], 'k')
            plt.plot([N-1, N-i-1], [i, 0], 'k')
    X = []
    colors = []
    sizes = []
    if not draw_nodes:
        return
    
    for i, n in enumerate(nodes):
        sz = 50
        if n.visited:
            sz = 100
        c = [0, 0.5, 0]
        if n.touched:
            if n.visited:
                c = [0, 0, 0]
            else:
                c = [1, 0, 0]
        colors.append(c)
        sizes.append(sz)
        X.append([n.data['x'], n.data['y']])
    X = np.array(X)
    colors = np.array(colors)
    sizes = np.array(sizes)
    plt.scatter(X[:, 0], X[:, 1], s=sizes, c=colors, zorder=10)
    plt.axis("off")
                

def make_delaunay_graph(N):
    x = np.random.rand(N)
    y = np.random.rand(N)
    nodes = []
    for i in range(N):
        n = GraphNode(i)
        n.data = {'x':x[i], 'y':y[i]}
        nodes.append(n)
    tri = Delaunay(np.array([x, y]).T).simplices
    edges = set()
    for i in range(tri.shape[0]):
        for k in range(3):
            i1, i2 = tri[i, k], tri[i, (k+1)%3]
            nodes[i1].edges.append(nodes[i2])
            nodes[i2].edges.append(nodes[i1])
            d = np.sqrt(np.sum((x[i1]-x[i2])**2 + (y[i1]-y[i2])**2))
            edges.add((i1, i2, d))
    return nodes, list(edges)


def make_grid_graph(N, seed = 0, include_diags=True):
    """
    Parameters
    ----------
    N: int
        Resolution of grid
    """
    np.random.seed(seed)
    nodes = []
    for i in range(N):
        for j in range(N):
            n = GraphNode(i*N+j)
            n.data = {'x':j, 'y':i}
            nodes.append(n)
    edges = []
    neighbs = []
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di != 0 or dj != 0:
                if include_diags or abs(di)+abs(dj) < 2:
                    neighbs.append([di, dj])
    for i in range(N):
        for j in range(N):
            idx1 = i*N + j
            for [di, dj] in neighbs:
                ii = i + di
                jj = j + dj
                if ii >= 0 and jj >= 0 and ii < N and jj < N:
                    idx2 = ii*N + jj
                    x1, y1 = nodes[idx1].data['x'], nodes[idx1].data['y']
                    x2, y2 = nodes[idx2].data['x'], nodes[idx2].data['y']
                    dx = x1-x2
                    dy = y1-y2
                    d = (dx**2 + dy**2)**0.5
                    edges.append((idx1, idx2, d))
                    nodes[idx1].edges.append(nodes[idx2])
                    nodes[idx1].dists.append(d)
                    nodes[idx2].edges.append(nodes[idx1])
                    nodes[idx2].dists.append(d)
    return nodes, edges


def get_collection_str(coll):
    s = "["
    for i, n in enumerate(list(coll)):
        if (i+1) % 20 == 0:
            s += "\n"
        s += "%i"%n.idx
        if i < len(coll)-1:
            s += ","
    s += "]"
    return s

def dist_of_edge(e):
    return e[2]

def get_mst_kruskal(nodes, edges):
    from unionfind import UFFast
    edges = sorted(edges, key = dist_of_edge)
    djset = UFFast(len(nodes))
    new_edges = []
    for e in edges:
        (i, j, d) = e
        if not djset.find(i, j):
            djset.union(i, j)
            new_edges.append(e)
    # Update node adjacency
    for n in nodes:
        n.edges = []
        n.dists = []
    for (i, j, d) in new_edges:
        nodes[i].edges.append(nodes[j])
        nodes[i].dists.append(d)
        nodes[j].edges.append(nodes[i])
        nodes[i].dists.append(d)
    return new_edges


def do_bfs(nodes, edges, draw_labels=True, plot_fn=draw_2d_graph):
    queue = deque()
    nodes[0].touched = True
    queue.append(nodes[0])
    plt.figure(figsize=(8, 6))
    idx = 0
    while len(queue) > 0:
        plt.clf()
        plot_fn(nodes, edges, draw_labels=draw_labels)
        n = queue.popleft()
        queue.appendleft(n)
        if draw_labels:
            plt.title("Processing {}\n{}".format(n.idx, get_collection_str(queue)))
        plt.savefig("{}.png".format(idx), bbox_inches='tight')
        n = queue.popleft()
        n.visited = True
        for e in n.edges:
            if not e.visited and not e.touched:
                e.touched = True
                queue.append(e)
        idx += 1
    plt.clf()
    plot_fn(nodes, edges, draw_labels=draw_labels)
    plt.savefig("{}.png".format(idx), bbox_inches='tight')


def do_dijkstra(nodes, edges, draw_labels=True, plot_fn=draw_2d_graph):
    for n in nodes:
        n.touched = False
        n.visited = False
    queue = HeapTree(lambda entry: entry[1].idx)
    nodes[0].touched = True
    queue.push((0, nodes[0]))
    plt.figure(figsize=(8, 6))
    idx = 0
    while len(queue) > 0:
        (d, n) = queue.peek()
        if n.visited:
            queue.pop()
            continue
        plt.clf()
        plot_fn(nodes, edges, draw_labels=draw_labels)
        if draw_labels:
            s = get_collection_str([n[1].idx for n in queue._arr])
            plt.title("Processing {} at distance {}\n{}".format(n.idx, d, s))
        else:
            plt.title("Distance {:.3f}, {} Nodes on Heap".format(d, len(queue)))
        plt.savefig("{}.png".format(idx), bbox_inches='tight')
        (d, n) = queue.pop()
        n.visited = True
        for i in range(len(n.edges)):
            ni = n.edges[i]
            if not ni.visited:
                ni.touched = True
                di = n.dists[i]
                queue.push((d+di, ni))
        idx += 1
    plt.clf()
    plot_fn(nodes, edges, draw_labels=draw_labels)
    plt.savefig("{}.png".format(idx), bbox_inches='tight')


def do_dfs(nodes, edges, draw_labels=True, plot_fn=draw_2d_graph):
    stack = [nodes[0]]
    nodes[0].touched = True
    plt.figure(figsize=(8, 6))
    idx = 0
    while len(stack) > 0:
        plt.clf()
        plot_fn(nodes, edges, draw_labels=draw_labels)
        n = stack[-1]
        if draw_labels:
            plt.title("Processing {}\n{}".format(n.idx, get_collection_str(stack)))
        plt.savefig("{}.png".format(idx), bbox_inches='tight')
        n = stack.pop()
        n.visited = True
        for e in n.edges:
            if not e.visited and not e.touched:
                e.touched = True
                stack.append(e)
        idx += 1
    plt.clf()
    plot_fn(nodes, edges, draw_labels=draw_labels)
    plt.savefig("{}.png".format(idx), bbox_inches='tight')

def do_2d_tsp(nodes, edges, start_node = 0, do_plot=True, draw_labels=True):
    """
    Do a depth-first search on the MST
    """
    for n in nodes:
        n.touched = False
        n.visited = False
    edges = get_mst_kruskal(nodes, edges)
    stack = [nodes[start_node]]
    nodes[start_node].touched = True
    if do_plot:
        plt.figure(figsize=(12, 6))
    idx = 0
    path = []
    tsp_edges = []
    total_len = 0
    plot_fn=draw_2d_graph
    while len(stack) > 0:
        n = stack[-1]
        path.append(n)
        if len(path) > 1:
            [n1, n2] = path[-2::]
            dx = n1.data['x'] - n2.data['x']
            dy = n1.data['y'] - n2.data['y']
            d = (dx**2 + dy**2)**0.5
            tsp_edges.append((path[-2].idx, path[-1].idx, d))
            total_len += d
        if do_plot:
            plt.clf()
            plt.subplot(121)
            plot_fn(nodes, edges, draw_labels=draw_labels)
            if draw_labels:
                plt.title("Processing {}\n{}".format(n.idx, get_collection_str(stack)))
            plt.subplot(122)
            plot_fn(nodes, tsp_edges, draw_labels=False)
            plt.title("Total length = {:.3f}".format(total_len))
            plt.savefig("{}.png".format(idx), bbox_inches='tight')
        n = stack.pop()
        n.visited = True
        
        for e in n.edges:
            if not e.visited and not e.touched:
                e.touched = True
                stack.append(e)
        idx += 1
    
    # Loop path back around
    path.append(path[0])
    [n1, n2] = path[-2::]
    dx = n1.data['x'] - n2.data['x']
    dy = n1.data['y'] - n2.data['y']
    d = (dx**2 + dy**2)**0.5
    tsp_edges.append((path[-2].idx, path[-1].idx, d))
    total_len += d

    if do_plot:
        plt.clf()
        plt.subplot(121)
        plot_fn(nodes, edges, draw_labels=draw_labels)
        plt.subplot(122)
        plot_fn(nodes, tsp_edges, draw_labels=False)
        plt.title("Total length = {:.3f}".format(total_len))
        plt.savefig("{}.png".format(idx), bbox_inches='tight')
    
    return path, total_len


if __name__ == '__main__':
    """
    np.random.seed(1)
    nodes, edges = make_delaunay_graph(20)
    do_2d_tsp(nodes, edges, 11)
    """

    #"""
    np.random.seed(0)
    N = 20
    nodes, edges = make_grid_graph(N)
    idx = N*int(N/2) + int(N/2)
    nodes[0], nodes[idx] = nodes[idx], nodes[0]
    do_dijkstra(nodes, edges, draw_labels=False, plot_fn=draw_grid_graph)
    #"""

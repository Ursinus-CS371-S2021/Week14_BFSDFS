import numpy as np
import matplotlib.pyplot as plt

class HeapTree(object):
    def __init__(self):
        self._arr = []

    def __len__(self):
        return len(self._arr)

    def _children(self, i):
        """
        Return the indices of the children of the node at index i

        Parameters
        ----------
        i: int
            Index of the element
        
        Returns
        -------
        list of ints: Indices of children.  There are 0, 1, or 2
        """
        children = []
        if 2*i+1 < len(self._arr):
            children.append(2*i+1)
        if 2*i+2 < len(self._arr):
            children.append(2*i+2)
        return children
    
    def _parent(self, i):
        """
        Return the index of the parent of this node

        Parameters
        ----------
        i: int
            Index of node
        
        Returns
        -------
        int: index of parent
        """
        return (i-1)//2
    
    def _swap(self, i, j):
        """
        Swap two nodes in the underlying array

        Parameters
        ----------
        i: int
            Index of first node
        j: int
            Index of second node
        """
        self._arr[i], self._arr[j] = self._arr[j], self._arr[i]

    def _upheap(self, i):
        """
        Recursively bubble a node up the heap if the node above
        it is greater

        Parameters
        ----------
        i: int
            Index of node to bubble up
        """
        parent = self._parent(i)
        if i > 0 and self._arr[i][0] < self._arr[parent][0]:
            self._swap(i, parent)
            self._upheap(parent)
    
    def _downheap(self, i):
        """
        Recursively bubble a node down the heap while it's greater 
        than one of its children

        Parameters
        ----------
        i: int
            Index of node to bubble down
        """
        children = self._children(i)
        if len(children) > 0:
            child = children[0]
            if len(children) > 1:
                c2 = children[1]
                if self._arr[c2][0] < self._arr[child][0]:
                    child = c2
            if self._arr[child][0] < self._arr[i][0]:
                self._swap(i, child)
                self._downheap(child)

    def push(self, entry):
        """
        Put a new value onto the heap

        Parameters
        ----------
        entry: tuple (float, ...)
            A tuple whose first entry is the priority (the rest is ignored)
        """
        self._arr.append(entry)
        self._upheap(len(self._arr)-1)
    
    def peek(self):
        """
        Return the value with the highest priority from the heap

        Returns
        -------
        tuple (float, ...)
        """
        assert(len(self) > 0)
        return self._arr[0]

    def pop(self):
        """
        Remove the value with the highest priority from the heap and
        return it

        Returns
        -------
        tuple (float, ...)
        """
        assert(len(self) > 0)
        ret = self._arr[0]
        # Move the last element to the root and bubble
        # down if necessary
        self._arr[0] = self._arr[-1]
        self._arr.pop()
        self._downheap(0)
        return ret


    def draw(self):
        """
        Draw the heap in its current state
        """
        N = len(self._arr)
        height = int(np.ceil(np.log2(N)))
        width = 2**height
        xs = np.zeros(N)
        ys = np.zeros(N)
        level = -1
        xi = 0
        # First draw nodes, and remember positions
        # in the process
        x0 = width/2
        for i in range(N):
            if np.log2(i+1) == int(np.log2(i+1)):
                level += 1
                xi = 0
                x0 -= 2**(height-level-1)
            stride = 2**(height-level)
            x = x0 + xi*stride
            y = -5*level
            plt.scatter([x], [y], 100, c='k')
            s = "{}".format(self._arr[i][0])
            if self._arr[i][1]:
                s = s + " ({})".format(self._arr[i][1])
            plt.text(x+0.5, y, s)
            xs[i] = x
            ys[i] = y
            xi += 1
        # Next draw edges
        for i in range(N):
            for j in self._children(i):
                plt.plot([xs[i], xs[j]], [ys[i], ys[j]])
        plt.axis("off")
        plt.axis("equal")

if __name__ == '__main__':
    T = HeapTree()
    np.random.seed(0)
    vals = np.random.permutation(100)
    print(vals)
    for v in vals:
        T.push((v, None))

    T.draw()
    plt.show()

    while len(T) > 0:
        print(T.pop()[0], end = ' ')
    print("")

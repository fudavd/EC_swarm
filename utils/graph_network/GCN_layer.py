import numpy as np
from scipy.linalg import sqrtm

class GCNLayerNumpy:
    def __init__(self, in_features:int, out_features:int, filter_number:int, self_connections:bool, init_weight:bool, reservoir:bool=True, activation=None, name:str="") -> None:
        """
        Implementation of a GCN layer in numpy

        The layer is defined as:
        Z = D^(-1/2) * A * D^(-1/2) * X * W

        where D is the diagonal matrix of the sum of the adjacency matrix
        and the identity matrix, A is the adjacency matrix, X is the node
        features matrix and W is the weight matrix.

        if self_connections is True, the adjacency matrix is modified
        by adding the identity matrix to it.

            :param in_features: Number of input features
            :param out_features: Number of output features
            :param self_connections: Whether to include self connections
            :param activation: Activation function
            :param name: Name of the layer
        """
        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number
        self.self_connections = self_connections
        self.init_weight = init_weight
        self.reservoir = reservoir
        if init_weight:
            self.W = self.glorot_init(self.in_features, self.out_features)
        if self.reservoir:
            self.W = np.random.uniform(-1, 1, (self.in_features, self.out_features))
        self.activation = activation
        self.name = name

    def __repr__(self):
        return f"""GCN Layer {self.name if self.name else ''}:
        W{'_' + self.name if self.name else ''} ({self.in_features}, {self.out_features})
        Activation: {self.activation}
        Self connections: {self.self_connections}
        K-hops: {self.filter_number}
        """
    
    def set_weights(self, W):

        assert W.shape == (self.in_features, self.out_features), "Weight matrix shape mismatch"
        self.W = W

    def forward(self, adj_matrix, node_feats, W=None, b=None):
        
        """
        Forward pass of the GCN layer. Can accept batch size.

        :param adj_matrix: Adjacency matrix, shape (batch_size, n_nodes, n_nodes)
        :param node_feats: Node features, shape (batch_size, n_nodes, in_features)
        :param W: Weight matrix, shape (in_features, out_features)
        :param b: Bias vector, shape (out_features)
        
        :return: Node features, shape (batch_size, n_nodes, out_features)
        """
        using_batch = (len(adj_matrix.shape) == 3 )
        if using_batch:
            raise "Batch dimension not implemented yet"
            assert adj_matrix.shape[0] == node_feats.shape[0], "Batch size mismatch, adjacency matrix and node features must have the same batch size"
            assert adj_matrix.shape[1] == node_feats.shape[1], "Number of nodes mismatch, adjacency matrix and node features must have the same number of nodes"
            assert adj_matrix.shape[1] == adj_matrix.shape[2], "Adjacency matrix must be square"
            assert node_feats.shape[2] == self.in_features, "Number of input features mismatch, node features must have the same number of features as the layer"

            n_nodes = adj_matrix.shape[1]
            batch_size = adj_matrix.shape[0]

        else:
            assert adj_matrix.shape[0] == adj_matrix.shape[1], "Adjacency matrix must be square"
            assert adj_matrix.shape[0] == node_feats.shape[0], "Number of nodes mismatch, adjacency matrix and node features must have the same number of nodes"
            assert node_feats.shape[1] == self.in_features, "Number of input features mismatch, node features must have the same number of features as the layer"

            n_nodes = adj_matrix.shape[0]
            # batch_size = 1
            # adj_matrix = adj_matrix.reshape([1, n_nodes, n_nodes])
            # node_feats = node_feats.reshape([1, n_nodes, self.in_features])

        if W is None and self.init_weight:
            raise "No weight matrix provided"
        if W is None and not self.init_weight:
            W = self.W
        
        if self.self_connections:
            adj_matrix += np.eye(n_nodes)


        # Adjacency matrix normalization || Laplacian matrix normalization

        ## Generating an empty D
        D_mod = np.zeros_like(adj_matrix)
        D_mod_invroot = np.zeros_like(D_mod)

        ## Filling it with the total number of neigh (plus self connections)
        
        # if using_batch:
        #     for i in range(batch_size):
        #         np.fill_diagonal(D_mod[i,], np.asarray(adj_matrix[i].sum(axis=0)))
        #         D_mod_invroot[i] = np.linalg.inv(sqrtm(D_mod[i]))

        # else:
        np.fill_diagonal(D_mod, np.asarray(adj_matrix.sum(axis=1)))
        D_mod_invroot = np.linalg.inv(sqrtm(D_mod))         
        ## Normalizing the adjacency matrix
        adj_matrix = D_mod_invroot @ adj_matrix @ D_mod_invroot

        # Computing K hops 
        # node_feats = node_feats.reshape([self.in_features,n_nodes])
        for _ in range(0, self.filter_number):
            node_feats = adj_matrix @ node_feats
                  
        node_feats = node_feats @ W

        if b is not None:
            node_feats += node_feats + b

        if self.activation is not None:
            node_feats = self.activation(node_feats)
        return node_feats


    def glorot_init(self, nin, nout):
        sd = np.sqrt(6.0 / (nin + nout))
        return np.random.uniform(-sd, sd, size=(nin, nout))

def test():
    """
    Function to test the GCN layer
    """
    node_feats = np.arange(8, dtype=np.float32).reshape((4, 2))
    print(f"Node Features: \n{node_feats}\n Node Features Shape: {node_feats.shape}\n")

    adj_matrix = np.array([
          [1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]
        ],
        dtype=np.float32,
    )
    print(f"Adjacency Matrix: \n{adj_matrix}\n Adjacency Matrix Shape: {adj_matrix.shape}\n")
    n_nodes = adj_matrix.shape[1]
    in_features = node_feats.shape[1]
    out_features = 2
    K = 2
    gcn = GCNLayerNumpy(
        in_features,
        out_features,
        self_connections=True,
        init_weight=False,
        filter_number=K,
        name="test",
    )
    print(gcn)
    W = np.array([[1, 0], [0., 1]])
    print(f"W: \n{W}\n W Shape: {W.shape}\n")
    node_feats_t1 = gcn.forward(adj_matrix, node_feats, W=W)
    print(f"Node Features T1: \n{node_feats_t1}\n Node Features T1 Shape: {node_feats_t1.shape}\n")

if __name__ == "__main__":
    test()
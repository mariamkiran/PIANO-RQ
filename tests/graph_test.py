
import networkx as nx


#graph details to help it select sub graph
#input_graph_file="../graph_datasets/DBLP-new.txt"


def max_degree_nodes(input_graph):
    """
    Finds and returns a list of nodes with the maximum degree in the graph.

    Args:
        graph: A networkx graph object.

    Returns:
        A list of nodes with the maximum degree. Returns 0 if the graph is empty and -1 if the graph is disconnected.
    """
    if not graph.nodes:
        return 0
    if not nx.is_connected(graph):
        return -1

    max_degree_value = max(dict(graph.degree()).values())
    return [node for node, degree in graph.degree() if degree == max_degree_value]

def build_graph_from_txt(input_graph):
    graph = nx.Graph()
    with open(input_graph, 'r') as file:
        for line in file:
            nodes = line.strip().split()
            print(nodes)
            if len(nodes) == 2:  # Ensure there are two nodes per line
                node1, node2 = nodes
                graph.add_edge(node1, node2)
                print(node1, " + ", node2)
            else: #assuming first two columns are nodes
                node1=nodes[0]
                node2=nodes[1]
                graph.add_edge(node1, node2)
                print(node1, " + ", node2)
            
    return graph

def main():
    graph=build_graph_from_txt(input_graph_file)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")

main

import networkx as nx
from matplotlib import pyplot as plt


def graph(closest: list):
    plt.subplots(figsize=(14, 14))

    G = nx.Graph()

    for node in closest:

        node_key = node['key']
        close_nodes = node['neighbors']

        G.add_node(node_key[1])

        for n in close_nodes:
            G.add_node(n[1], size=n[2])
            G.add_edge(node_key[1], n[1], weight=abs(node_key[0] - n[0]))

    node_sizes = nx.get_node_attributes(G, 'size')
    edge_weights = nx.get_edge_attributes(G, 'weight')

    pos = nx.spring_layout(G, k=0.5, iterations=20)

    # [TODO] Fazer com que o colormap mude de acordo com o crescimento do nó.
    # cmap = plt.cm.coolwarm
    # colors = [n for n in range(len(G.nodes()))]

    nx.draw(
        G,
        pos,
        with_labels=True,
        font_size=12,
        font_weight='bold',
        node_size=[n * 1.5 * 10 ** 5 for n in node_sizes.values()],
        width=list(edge_weights.values()),
        node_color='red',
        edge_color='grey',
        alpha=1
    )
    plt.show()

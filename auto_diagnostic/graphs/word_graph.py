import networkx as nx
from matplotlib import pyplot as plt
from dataloader.makedata import Dataloader

from auto_diagnostic.tfidf import TFIDF
from auto_diagnostic.lemmatization import Lemmatization
from auto_diagnostic.preprocess import tokenize


def graph(closest: list):
    fig, ax = plt.subplots(figsize=(14, 14))

    G = nx.Graph()

    for node in closest:

        node_key = node['key']
        close_nodes = node['neighbors']

        G.add_node(node_key[1], size=node_key[2])

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
        font_size=10,
        font_weight='bold',
        node_size=[n * 10 ** 5 for n in node_sizes.values()],
        width=list(edge_weights.values()),
        node_color='red',
        edge_color='grey',
        alpha=1
    )
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000") 
    plt.show()


if __name__ == '__main__':
    dataloader = Dataloader()
    df = dataloader.make_csv()
    tfidf = TFIDF()
    tokenized = tokenize(df['text_column'])
    model = Lemmatization()
    lemmatized_list = model.lemmatize(tokenized)
    _ = tfidf.get_wordset_from_text(lemmatized_list)
    tfidf_vectors = tfidf.tf_idf()
    closest = tfidf.get_closest_neighbors(vectors=tfidf_vectors)
    graph(closest)
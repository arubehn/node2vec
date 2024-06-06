import networkx as nx
from node2vec.graph import Graph
from node2vec.train import learn_embeddings
from pyconcepticon import Concepticon
from pathlib import Path
from itertools import combinations


# NODE2VEC PARAMETERS ###########
DIRECTED = False
P = 1
Q = 1
NUM_WALKS = 10
WALKS_LEN = 80
OUTPUT = Path(__file__).parent / "emb" / "colex.emb"
#################################

con = Concepticon()
clist = con.conceptlists["List-2023-1308"]

idxs = [c.id for c in clist.concepts.values()]
concepts = [clist.concepts[idx].concepticon_gloss for idx in idxs]
# matrix = [[0.00001 for c in concepts] for c in concepts]

print("Setting up graph...")

graph = nx.Graph()

visited = set()
for i, (idx, concept) in enumerate(zip(idxs, concepts)):
    # iterate over all links and fill the matrix
    if concept not in graph:
        graph.add_node(concept)
    for node in clist.concepts[idx].attributes["linked_concepts"]:
        concept_b = node["NAME"]
        if concept_b in graph:
            graph.add_node(concept_b)
        if (concept, concept_b) not in visited:
            visited.add((concept, concept_b))
            visited.add((concept_b, concept))
            graph.add_edge(concept, concept_b, weight=node["FullFams"])

print("Done.")
print("Smoothing graph...")

# add smoothing to avoid 0-division
nodes = list(graph.nodes)
for x, y in combinations(nodes, 2):
    edge_data = graph.get_edge_data(x, y)
    weight = edge_data["weight"] if edge_data else 0
    graph.add_edge(x, y, weight=weight+1)

print("Done.")

# adjusted "main" method
gr = Graph(graph, DIRECTED, P, Q)
print("Preprocessing transition probabilities...")
gr.preprocess_transition_probs()
print("Done.")

print("Simulating walks...")
walks = gr.simulate_walks(NUM_WALKS, WALKS_LEN)
print("Done.")

print("Training embeddings...")
learn_embeddings(walks, output=OUTPUT)
print("Done.")

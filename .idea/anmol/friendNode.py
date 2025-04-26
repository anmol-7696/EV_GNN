import networkx as nx
import matplotlib.pyplot as plt 

# # Create an empty undirected graph
# social_network = nx.Graph()

# # Add nodes representing users
# users = [1, 2, 3, 4]
# social_network.add_nodes_from(users)

# # Add edges representing friendships
# friendships = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
# social_network.add_edges_from(friendships)

social_network = nx.Graph()
users = [1, 2, 3, 4]
social_network.add_nodes_from(users)

friendship = [(1,2) , (1,3) , (1,4) , (2,3) , (2,4), (3,4)]
social_network.add_edges_from(friendship)

# Visualize the social network graph
pos = nx.spring_layout(social_network)  # Positions for all nodes
nx.draw(social_network, pos, with_labels=True, node_color='skyblue', node_size=1000,
font_size=12, font_weight='bold')
plt.title("Social Network Graph")
plt.show()
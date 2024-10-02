import matplotlib.pyplot as plt
import numpy as np
import heapq
from matplotlib.animation import FuncAnimation, PillowWriter

# Function to load coordinates from file
def loadCoords(filename):
    coords = []
    with open(filename, 'r') as file:
        for line in file:
            x, y = map(float, line.split())
            coords.append((x, y))  # Store as tuple of floats
    return coords

# Function to load graph data from file (number of nodes, start, end, edges)
def loadGraph(filename):
    with open(filename, 'r') as file:
        numNodes = int(file.readline().strip())
        startNode = int(file.readline().strip())
        endNode = int(file.readline().strip())
        
        edges = []
        for line in file:
            u, v, weight = map(float, line.split())
            edges.append((int(u) - 1, int(v) - 1, weight))  # Convert to zero-indexed
        
    return numNodes, startNode - 1, endNode - 1, edges  # Return zero-indexed start/end

# Function to create an adjacency list from edges
def createAdjList(numNodes, edges):
    graph = {i: [] for i in range(numNodes)}  # Initialize empty adjacency list
    for u, v, weight in edges:
        graph[u].append((v, weight))  # Add neighbor and weight
        graph[v].append((u, weight))  # Since the graph is undirected
    return graph

# Dijkstra's algorithm implementation with step tracking
def dijkstraWithSteps(graph, startNode, numNodes):
    dist = {i: float('inf') for i in range(numNodes)}  # Initialize distances to infinity
    dist[startNode] = 0  # Distance to start node is 0
    prev = {i: None for i in range(numNodes)}  # Previous nodes for path reconstruction
    priorityQueue = [(0, startNode)]  # Min-heap to track nodes by distance
    
    steps = []  # Keep track of each step in the algorithm
    
    while priorityQueue:
        currentDist, currentNode = heapq.heappop(priorityQueue)  # Get node with min distance
        
        # Skip if this distance is not better
        if currentDist > dist[currentNode]:
            continue
        
        # Record the current state of the algorithm
        steps.append((currentNode, dist.copy(), prev.copy()))
        
        # Explore neighbors of current node
        for neighbor, weight in graph[currentNode]:
            distance = currentDist + weight
            
            # If found a shorter path, update distance and predecessor
            if distance < dist[neighbor]:
                dist[neighbor] = distance
                prev[neighbor] = currentNode
                heapq.heappush(priorityQueue, (distance, neighbor))  # Push updated distance
    
    return dist, prev, steps

# Function to reconstruct the shortest path from start to end
def reconstructPath(prev, startNode, endNode):
    path = []
    currentNode = endNode
    while currentNode is not None:
        path.append(currentNode)
        currentNode = prev[currentNode]  # Go backwards through predecessors
    path.reverse()  # Reverse the list to get path from start to end
    return path if path[0] == startNode else []  # Return path only if valid

# Function to plot the graph and animate Dijkstra's steps
def plotAndAnimate(coords, graph, steps, startNode, endNode, finalPath, outputFile='dijkstra_animation.mp4'):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the graph nodes and edges
    ax.scatter(*zip(*coords), color='blue')  # Plot nodes
    for u, neighbors in graph.items():
        for v, _ in neighbors:
            ax.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]], 'gray', lw=0.5)  # Plot edges
    
    # Highlight the start and end nodes
    ax.scatter(*coords[startNode], color='green', s=100, label='Start Node')
    ax.scatter(*coords[endNode], color='red', s=100, label='End Node')
    
    # Initialize line collections for paths
    exploredLines, = ax.plot([], [], 'orange', lw=2, label='Explored Path')
    currentNodeDot, = ax.plot([], [], 'yo', markersize=15, label='Current Node')
    finalPathLines, = ax.plot([], [], 'red', lw=4, label='Final Path')
    
    def init():
        exploredLines.set_data([], [])
        currentNodeDot.set_data([], [])
        finalPathLines.set_data([], [])
        return exploredLines, currentNodeDot, finalPathLines

    # Update function for each animation step
    def update(step):
        currentNode, dist, prev = steps[step]
        
        # Prepare explored paths so far
        xdata, ydata = [], []
        for i in range(len(prev)):
            if prev[i] is not None:
                xdata.extend([coords[i][0], coords[prev[i]][0], None])
                ydata.extend([coords[i][1], coords[prev[i]][1], None])
        exploredLines.set_data(xdata, ydata)
        
        # Update current node dot (needs to be a sequence of length 1)
        currentNodeDot.set_data([coords[currentNode][0]], [coords[currentNode][1]])
        
        # For the last frame, show the final shortest path
        if step == len(steps) - 1:
            finalXdata, finalYdata = [], []
            for i in range(len(finalPath) - 1):
                u = finalPath[i]
                v = finalPath[i + 1]
                finalXdata.extend([coords[u][0], coords[v][0], None])
                finalYdata.extend([coords[u][1], coords[v][1], None])
            finalPathLines.set_data(finalXdata, finalYdata)
        
        return exploredLines, currentNodeDot, finalPathLines

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(steps), init_func=init, blit=True, repeat=False)
    
    # Save the animation as mp4
    ani.save(outputFile, writer='ffmpeg', fps=10)
    
    plt.close(fig)  # Close the figure to free memory

# Function to write the shortest path and distances to output.txt
def writeOutputFile(path, dist):
    with open('output.txt', 'w') as file:
        # Write node indices
        file.write(' '.join(str(node + 1) for node in path) + '\n')  # Convert back to 1-indexed
        # Write cumulative distances
        file.write(' '.join(f'{dist[node]:.1f}' for node in path) + '\n')

# Main function to run the entire process
def main():
    coordsFile = 'coords.txt'
    graphFile = 'input.txt'
    outputVideo = './dijkstraAnimation.mp4'
    
    # Load coordinates and graph
    coords = loadCoords(coordsFile)
    numNodes, startNode, endNode, edges = loadGraph(graphFile)
    graph = createAdjList(numNodes, edges)
    
    # Run Dijkstra's algorithm and track steps
    dist, prev, steps = dijkstraWithSteps(graph, startNode, numNodes)
    finalPath = reconstructPath(prev, startNode, endNode)  # Get the final shortest path
    
    # Write the shortest path and distances to output.txt
    writeOutputFile(finalPath, dist)
    
    # Create animation
    plotAndAnimate(coords, graph, steps, startNode, endNode, finalPath, outputFile=outputVideo)
    print(f'Video generated: {outputVideo}')

# Uncomment to run the main function
if __name__ == "__main__":
    main()



def draw_save_mermaid_png(graph, png_path: str="mermaid_diagram.png"):
    try:
        png_data = graph.get_graph().draw_mermaid_png() 
        with open(png_path, "wb") as f:
            f.write(png_data)
    except Exception as e:
        print(f"Error drawing mermaid png file: {e}")

"""
In the graph: 
1. Solid edges represent a direct, deterministic flow. This means the process moves from one node to the next in a straightforward, required sequence without conditional branching or optional steps.
2. Dashed edge often signifies a conditional, optional, or less strict transition.
"""
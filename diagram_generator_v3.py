# """
# diagram_generator.py

# Takes a structured architecture plan (components + connections)
# and renders a Graphviz diagram to a PNG image.

# Also returns the Graphviz DOT source so the user can edit it
# or load it into other tools.
# """

# import os
# from typing import Dict, Any, Tuple
# from uuid import uuid4
# from graphviz import Digraph

# # Manually add Graphviz bin folder (no need to add to system PATH)
# os.environ["PATH"] += os.pathsep + r"C:\graphviz-14.0.4\bin"

# def ensure_output_dir() -> str:
#     """
#     Ensure the 'static/diagrams' directory exists.
#     Flask will serve files from the 'static' folder by default.
#     """
#     output_dir = os.path.join("static", "diagrams")
#     os.makedirs(output_dir, exist_ok=True)
#     return output_dir


# def generate_graphviz_diagram(arch_plan):
#     components = arch_plan.get("components", [])
#     connections = arch_plan.get("connections", [])

#     # Create the graph
#     dot = Digraph(comment="Architecture Diagram")

#     # Layout tuning for cleaner look
#     dot.attr(
#         "graph",
#         rankdir="LR",      # left → right
#         splines="ortho",   # orthogonal, right-angle edges
#         concentrate="true",# merge parallel edges where possible
#         nodesep="0.6",
#         ranksep="0.9",
#     )

#     dot.attr(
#         "node",
#         shape="box",
#         style="rounded,filled",
#         fontsize="12",
#     )
#     dot.attr("edge", fontsize="9")

#     # --- Group components by type ---
#     layers = {
#         "frontend": [],
#         "gateway": [],
#         "services": [],
#         "databases": [],
#         "pipeline": [],
#         "other": [],
#     }

#     for c in components:
#         ctype = (c.get("type") or "").lower()
#         if ctype in ("client", "web"):
#             layers["frontend"].append(c)
#         elif ctype in ("gateway",):
#             layers["gateway"].append(c)
#         elif ctype in ("app", "service", "microservice"):
#             layers["services"].append(c)
#         elif ctype in ("database", "db"):
#             layers["databases"].append(c)
#         elif ctype in ("data_pipeline", "pipeline", "etl"):
#             layers["pipeline"].append(c)
#         else:
#             layers["other"].append(c)

#     # Helper to add a cluster if it has nodes
#     def add_cluster(name, label, comps):
#         if not comps:
#             return
#         with dot.subgraph(name=name) as sg:
#             sg.attr(label=label, style="rounded,dashed", color="#cccccc")
#             for c in comps:
#                 sg.node(c["id"], c["label"])

#     # --- Create visual clusters ---
#     add_cluster("cluster_frontend", "Frontend", layers["frontend"])
#     add_cluster("cluster_gateway", "API Gateway", layers["gateway"])
#     add_cluster("cluster_services", "Services", layers["services"])
#     add_cluster("cluster_databases", "Databases", layers["databases"])
#     add_cluster("cluster_pipeline", "Reporting / Data Pipeline", layers["pipeline"])
#     add_cluster("cluster_other", "Other", layers["other"])

#     # --- Draw edges ---
#     for conn in connections:
#         src = conn.get("from")
#         dst = conn.get("to")
#         if not src or not dst:
#             continue

#         label = conn.get("label") or ""

#         # Option A: keep labels (a bit busier but informative)
#         dot.edge(src, dst, label=label)

#         # Option B (cleaner): hide labels
#         # dot.edge(src, dst)

#     # --- Output SVG ---
#     output_dir = os.path.join("static", "diagrams")
#     os.makedirs(output_dir, exist_ok=True)

#     file_id = uuid4().hex
#     filename = f"arch_{file_id}"
#     filepath = os.path.join(output_dir, filename)

#     dot.format = "svg"
#     rendered_path = dot.render(filename=filepath, cleanup=True)

#     # Convert filesystem path to Flask static URL
#     relative_path = rendered_path.replace("\\", "/")
#     dot_source = dot.source

#     return "/" + relative_path, dot_source

"""
diagram_generator_v3.py

UPDATED FOR MULTI-AGENT POC

Takes structured architecture plan (components + connections)
and renders Graphviz diagram with support for multi-agent data.

Enhanced features:
- Better color coding for agent-generated components
- Support for agent identification in diagram
- SVG format for web display
- Error handling for missing Graphviz
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional
from uuid import uuid4

try:
    from graphviz import Digraph
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logging.warning("Graphviz not installed. Diagram generation disabled.")

# Manually add Graphviz bin folder (if on Windows)
if os.name == 'nt':  # Windows
    graphviz_path = r"C:\graphviz-14.0.4\bin"
    if os.path.exists(graphviz_path):
        os.environ["PATH"] += os.pathsep + graphviz_path

logger = logging.getLogger(__name__)


def ensure_output_dir() -> str:
    """
    Ensure the 'static/diagrams' directory exists.
    Flask serves files from 'static' folder by default.
    """
    output_dir = os.path.join("static", "diagrams")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_component_color(component_type: str) -> str:
    """
    Return color for component type.
    
    Color coding:
    - Frontend: Blue
    - Gateway: Purple
    - Services: Green
    - Database: Orange
    - Cache: Red
    - Pipeline: Yellow
    - Other: Gray
    """
    ctype = (component_type or "").lower()
    
    colors = {
        "client": "#87CEEB",          # Sky blue
        "web": "#87CEEB",
        "gateway": "#DDA0DD",         # Plum
        "api": "#DDA0DD",
        "service": "#90EE90",         # Light green
        "app": "#90EE90",
        "microservice": "#90EE90",
        "database": "#FFB347",        # Pastel orange
        "db": "#FFB347",
        "cache": "#FFB6C1",           # Light pink
        "redis": "#FFB6C1",
        "memcached": "#FFB6C1",
        "queue": "#FFFFE0",           # Light yellow
        "message": "#FFFFE0",
        "kafka": "#FFFFE0",
        "pipeline": "#F0E68C",        # Khaki
        "etl": "#F0E68C",
        "monitoring": "#D3D3D3",      # Light gray
        "logging": "#D3D3D3",
    }
    
    return colors.get(ctype, "#E8E8E8")  # Default light gray


def generate_graphviz_diagram(arch_plan: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generate Graphviz diagram from architecture plan.
    
    Args:
        arch_plan: Dictionary with 'components' and 'connections'
        
    Returns:
        Tuple of (diagram_url, dot_source)
    """
    
    if not GRAPHVIZ_AVAILABLE:
        logger.error("Graphviz not available. Install with: pip install graphviz")
        return "", ""
    
    try:
        components = arch_plan.get("components", [])
        connections = arch_plan.get("connections", [])
        
        # Create the graph
        dot = Digraph(comment="Multi-Agent Generated Architecture Diagram")
        
        # Layout tuning for clean look
        dot.attr(
            "graph",
            rankdir="LR",              # Left to right
            splines="ortho",           # Orthogonal edges
            concentrate="true",        # Merge parallel edges
            nodesep="0.6",
            ranksep="0.9",
            bgcolor="white",
        )
        
        # Node styling
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fontsize="12",
            fontname="Arial",
            penwidth="2",
        )
        
        # Edge styling
        dot.attr(
            "edge",
            fontsize="9",
            fontname="Arial",
            color="#555555",
        )
        
        # --- Group components by type (for visual organization) ---
        layers = {
            "frontend": [],
            "gateway": [],
            "services": [],
            "databases": [],
            "cache": [],
            "queue": [],
            "pipeline": [],
            "monitoring": [],
            "other": [],
        }
        
        for c in components:
            ctype = (c.get("type") or "").lower()
            
            if ctype in ("client", "web", "frontend"):
                layers["frontend"].append(c)
            elif ctype in ("gateway", "api"):
                layers["gateway"].append(c)
            elif ctype in ("service", "app", "microservice"):
                layers["services"].append(c)
            elif ctype in ("database", "db"):
                layers["databases"].append(c)
            elif ctype in ("cache", "redis", "memcached"):
                layers["cache"].append(c)
            elif ctype in ("queue", "message", "kafka"):
                layers["queue"].append(c)
            elif ctype in ("pipeline", "etl", "data_pipeline"):
                layers["pipeline"].append(c)
            elif ctype in ("monitoring", "logging"):
                layers["monitoring"].append(c)
            else:
                layers["other"].append(c)
        
        # Helper function to add a cluster
        def add_cluster(cluster_name: str, label: str, comps: list):
            """Add components in a visual cluster."""
            if not comps:
                return
            
            with dot.subgraph(name=f"cluster_{cluster_name}") as sg:
                sg.attr(
                    label=label,
                    style="rounded,dashed",
                    color="#CCCCCC",
                    fontname="Arial",
                )
                
                for c in comps:
                    node_id = c.get("id", "unknown")
                    node_label = c.get("label", node_id)
                    node_type = c.get("type", "other")
                    color = get_component_color(node_type)
                    
                    sg.node(
                        node_id,
                        node_label,
                        fillcolor=color,
                        color="#333333",
                    )
        
        # --- Create visual clusters (organized by layer) ---
        add_cluster("frontend", "🖥️ Frontend Layer", layers["frontend"])
        add_cluster("gateway", "🚪 API Gateway", layers["gateway"])
        add_cluster("services", "⚙️ Services", layers["services"])
        add_cluster("databases", "💾 Databases", layers["databases"])
        add_cluster("cache", "⚡ Cache Layer", layers["cache"])
        add_cluster("queue", "📨 Message Queue", layers["queue"])
        add_cluster("pipeline", "📊 Data Pipeline", layers["pipeline"])
        add_cluster("monitoring", "📈 Monitoring", layers["monitoring"])
        add_cluster("other", "🔧 Other", layers["other"])
        
        # --- Draw connections (edges) ---
        for conn in connections:
            src = conn.get("from")
            dst = conn.get("to")
            
            if not src or not dst:
                continue
            
            label = conn.get("label", "")
            protocol = conn.get("protocol", "")
            
            # Combine label and protocol
            edge_label = f"{label}\n{protocol}".strip()
            
            dot.edge(src, dst, label=edge_label, color="#666666")
        
        # --- Render to SVG ---
        output_dir = ensure_output_dir()
        file_id = uuid4().hex
        filename = f"arch_{file_id}"
        filepath = os.path.join(output_dir, filename)
        
        # Render as SVG for web display
        dot.format = "svg"
        rendered_path = dot.render(filename=filepath, cleanup=True)
        
        # Convert filesystem path to Flask static URL
        relative_path = rendered_path.replace("\\", "/")
        diagram_url = "/" + relative_path
        dot_source = dot.source
        
        logger.info(f"Generated diagram: {diagram_url}")
        
        return diagram_url, dot_source
        
    except Exception as e:
        logger.error(f"Error generating diagram: {e}")
        return "", ""


def generate_diagram_from_multi_agent_result(multi_agent_result: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generate diagram specifically from multi-agent orchestration result.
    
    Args:
        multi_agent_result: Result from MultiAgentOrchestrator
        
    Returns:
        Tuple of (diagram_url, dot_source)
    """
    
    try:
        # Extract architecture from multi-agent result
        architecture = multi_agent_result.get("architecture_design", {})
        
        if not architecture:
            logger.warning("No architecture found in multi-agent result")
            return "", ""
        
        # Generate diagram from architecture
        return generate_graphviz_diagram(architecture)
        
    except Exception as e:
        logger.error(f"Error generating diagram from multi-agent result: {e}")
        return "", ""


# For backward compatibility
def generate_diagram(arch_plan: Dict[str, Any]) -> Tuple[str, str]:
    """Backward compatibility wrapper."""
    return generate_graphviz_diagram(arch_plan)


if __name__ == "__main__":
    # Test function
    test_arch = {
        "components": [
            {"id": "frontend", "label": "React App", "type": "web"},
            {"id": "gateway", "label": "API Gateway", "type": "gateway"},
            {"id": "service1", "label": "User Service", "type": "service"},
            {"id": "service2", "label": "Order Service", "type": "service"},
            {"id": "db", "label": "PostgreSQL", "type": "database"},
            {"id": "cache", "label": "Redis", "type": "cache"},
            {"id": "queue", "label": "Kafka", "type": "queue"},
        ],
        "connections": [
            {"from": "frontend", "to": "gateway", "label": "HTTP", "protocol": "REST"},
            {"from": "gateway", "to": "service1", "label": "Route", "protocol": "gRPC"},
            {"from": "gateway", "to": "service2", "label": "Route", "protocol": "gRPC"},
            {"from": "service1", "to": "db", "label": "Query", "protocol": "SQL"},
            {"from": "service2", "to": "db", "label": "Query", "protocol": "SQL"},
            {"from": "service1", "to": "cache", "label": "Cache", "protocol": "Redis"},
            {"from": "service2", "to": "queue", "label": "Event", "protocol": "Message"},
        ]
    }
    
    url, source = generate_graphviz_diagram(test_arch)
    print(f"Diagram URL: {url}")
    print(f"DOT Source:\n{source}")
from flask import Flask, jsonify, render_template, request
import json
import os
from datetime import datetime
import asyncio

# NEW: Import multi-agent orchestrator
from multi_agent_poc import get_orchestrator
from diagram_generator_v3 import generate_diagram_from_multi_agent_result

import config

app = Flask(__name__)

# In-memory storage (for POC)
conversations = {}


@app.route("/")
def index():
    """Render main UI (same as before)."""
    return render_template("index_v3.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    MODIFIED ENDPOINT: Now uses MULTI-AGENT AUTONOMOUS ORCHESTRATION
    
    Instead of: user_input → single LLM → output
    Now does: user_input → [4 autonomous agents in parallel] → consolidated report
    
    Request:
    {
        "message": "user requirements",
        "conversation_id": "unique_id"
    }
    
    Response:
    {
        "success": true,
        "multi_agent_results": {
            "requirement_analysis": {...},
            "architecture_design": {...},
            "security_review": {...},
            "recommendations": {...}
        },
        "agent_execution_log": [...],  # SHOWS AGENT AUTONOMY
        "agents_involved": ["Agent1", "Agent2", ...],
        "total_confidence": 0.85
    }
    """
    
    try:
        data = request.get_json(force=True)
        user_message = (data.get("message") or "").strip()
        conversation_id = data.get("conversation_id") or "default"
        
        if not user_message:
            return jsonify({"error": "Message required"}), 400
        
        # Initialize conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                "history": [],
                "created_at": datetime.now().isoformat()
            }
        
        conv = conversations[conversation_id]
        conv["history"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # IMPORTANT: Use MULTI-AGENT ORCHESTRATOR instead of old agent
        print(f"\n{'='*60}")
        print(f"INVOKING MULTI-AGENT AUTONOMOUS ORCHESTRATION")
        print(f"Conversation: {conversation_id}")
        print(f"{'='*60}\n")
        
        orchestrator = get_orchestrator()
        
        # Run async orchestration (multi-agent)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        multi_agent_result = loop.run_until_complete(
            orchestrator.orchestrate_autonomously(
                requirements=user_message,
                conversation_id=conversation_id
            )
        )
        
        loop.close()
        
        if multi_agent_result.get("status") == "error":
            return jsonify({
                "success": False,
                "error": multi_agent_result.get("error")
            }), 500
        
        # Extract key findings from each agent (for UI display)
        architecture = multi_agent_result.get("architecture_design", {})
        security = multi_agent_result.get("security_review", {})
        recommendations = multi_agent_result.get("recommendations", {})
        
        # ========== FIXED: Generate diagram AFTER multi_agent_result is defined ==========
        diagram_url = ""
        dot_source = ""
        try:
            diagram_url, dot_source = generate_diagram_from_multi_agent_result(multi_agent_result)
            if diagram_url:
                print(f"✅ Diagram generated: {diagram_url}")
            else:
                print(f"⚠️  No diagram generated (Graphviz may not be installed)")
        except Exception as e:
            print(f"⚠️  Could not generate diagram: {e}")
        # ========== END OF FIX ==========
        
        # Build response showing MULTI-AGENT AUTONOMY
        response_payload = {
            "success": True,
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            
            # MULTI-AGENT RESULTS (shows they worked independently)
            "multi_agent_results": {
                "requirement_analysis": multi_agent_result.get("requirement_analysis", {}),
                "architecture_design": architecture,
                "security_review": security,
                "final_recommendations": recommendations
            },
            
            # Diagram section
            "diagram": {
                "url": diagram_url,
                "dot_source": dot_source,
                "generated": bool(diagram_url)
            },
            
            # AGENT EXECUTION LOG (SHOWS AUTONOMY)
            "agent_execution_log": multi_agent_result.get("agent_execution_log", []),
            
            # AGENTS INVOLVED (SHOWS MULTI-AGENT)
            "agents_involved": multi_agent_result.get("agents_involved", []),
            "total_confidence": multi_agent_result.get("total_confidence", 0.0),
            
            # For diagram generation (backward compatible)
            "architecture": {
                "components": architecture.get("components", []),
                "connections": architecture.get("connections", []),
                "summary": f"<h2>Multi-Agent Architecture</h2><p>Designed by {len(multi_agent_result.get('agents_involved', []))} autonomous agents</p>"
            },
            
            # Security findings
            "security_findings": {
                "threats_identified": len(security.get("threats", [])),
                "recommendations": security.get("security_recommendations", [])
            },
            
            # Show which agents contributed
            "autonomous_analysis": {
                "requirement_analyzer": "✅ Autonomous analysis complete",
                "architect": "✅ Autonomous design complete",
                "security_reviewer": "✅ Autonomous security review complete",
                "recommendation_engine": "✅ Autonomous consolidation complete"
            }
        }
        
        # Log the conversation
        conv["history"].append({
            "role": "assistant",
            "content": "Multi-agent autonomous analysis complete",
            "agents_used": len(multi_agent_result.get('agents_involved', [])),
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify(response_payload)
    
    except Exception as e:
        app.logger.error(f"Error in api_chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/agents/status", methods=["GET"])
def agents_status():
    """
    NEW ENDPOINT: Show agent status and their autonomous capabilities.
    
    This demonstrates MULTI-AGENT architecture:
    - Shows all 4 agents
    - Shows their autonomous capabilities
    - Shows their role specialization
    """
    
    return jsonify({
        "success": True,
        "multi_agent_system": {
            "total_agents": 4,
            "agents": [
                {
                    "name": "Requirement Analyzer",
                    "role": "Parse requirements autonomously",
                    "autonomous_capabilities": [
                        "Identifies hidden constraints",
                        "Recommends patterns without prompting",
                        "Identifies risks proactively"
                    ],
                    "status": "✅ Active"
                },
                {
                    "name": "Architecture Designer",
                    "role": "Design system autonomously",
                    "autonomous_capabilities": [
                        "Makes design decisions independently",
                        "Selects components without user guidance",
                        "Optimizes for constraints"
                    ],
                    "status": "✅ Active"
                },
                {
                    "name": "Security Reviewer",
                    "role": "Review security autonomously",
                    "autonomous_capabilities": [
                        "Analyzes threats WITHOUT user request",
                        "Recommends specific security measures",
                        "Validates compliance independently"
                    ],
                    "status": "✅ Active"
                },
                {
                    "name": "Recommendation Engine",
                    "role": "Consolidate findings autonomously",
                    "autonomous_capabilities": [
                        "Aggregates multi-agent findings",
                        "Resolves conflicts intelligently",
                        "Makes final recommendations"
                    ],
                    "status": "✅ Active"
                }
            ],
            "architecture_style": "Multi-Agent Autonomous System",
            "parallelization": "Agents execute in sequence for POC (can be parallel)",
            "key_features": [
                "Each agent has specialized expertise",
                "Autonomous decision-making (no user prompting needed)",
                "Agent reasoning transparency",
                "Consolidated multi-perspective analysis"
            ]
        }
    })


@app.route("/api/conversation/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    """Get conversation history."""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify({
        "success": True,
        "conversation": conversations[conversation_id]
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "3.0 (Multi-Agent POC)",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-AGENT ARCHITECTURE DESIGN ASSISTANT v3.0 (POC)")
    print("="*60)
    print("\nFeatures:")
    print("✅ 4 Autonomous Agents")
    print("✅ Multi-Agent Orchestration")
    print("✅ Specialized Agent Roles")
    print("✅ Autonomous Decision-Making")
    print("✅ Agent Reasoning Transparency")
    print("\nEndpoints:")
    print("  POST  /api/chat                 - Multi-agent analysis")
    print("  GET   /api/agents/status        - Show agent capabilities")
    print("  GET   /api/conversation/{id}/   - Conversation history")
    print("  GET   /health                   - System status")
    print("="*60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)
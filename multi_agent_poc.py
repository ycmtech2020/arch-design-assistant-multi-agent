"""
multi_agent_poc.py

LIGHTWEIGHT MULTI-AGENT POC (Proof of Concept)
For startup pitch - demonstrates autonomous agents with minimal complexity

Agents (4 core agents for POC):
1. RequirementAnalyzer - Parse requirements autonomously
2. ArchitectureDesigner - Design system autonomously  
3. SecurityReviewer - Review security autonomously
4. RecommendationEngine - Consolidate findings

This is NOT production-ready, just shows the concept working.
"""
import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import asyncio

import httpx
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage

import config

tiktoken_cache_dir = r"C:\Users\GenAIVARUSR3\Downloads\tiktoken_cache"
if not os.path.exists(tiktoken_cache_dir):
    os.makedirs(tiktoken_cache_dir)
logger = logging.getLogger(__name__)

# =========================== #
# AGENT RESULTS              #
# =========================== #

class AgentResult:
    """Simple result format from each agent."""
    
    def __init__(self, agent_name: str, findings: dict, reasoning: str, confidence: float):
        self.agent_name = agent_name
        self.findings = findings
        self.reasoning = reasoning
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()


# =========================== #
# LIGHTWEIGHT AGENTS (POC)   #
# =========================== #

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, name: str, temperature: float = 0.3):
        self.name = name
        self.temperature = temperature
        self.client = ChatOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            base_url=config.AZURE_OPENAI_ENDPOINT,
            model=config.OPENAI_MODEL,
            http_client=httpx.Client(verify=False),
            temperature=temperature,
            max_tokens=1500,
        )
    
    def get_system_prompt(self) -> str:
        """Override in subclasses."""
        raise NotImplementedError
    
    async def analyze(self, context: str) -> AgentResult:
        """Analyze and return findings."""
        raise NotImplementedError


class RequirementAnalyzerAgent(BaseAgent):
    """
    AUTONOMOUS Agent 1: Analyzes user requirements without user guidance.
    
    Shows AUTONOMY by:
    - Making intelligent inferences
    - Identifying hidden constraints  
    - Proactively suggesting patterns
    - Working WITHOUT follow-up prompts
    """
    
    def __init__(self):
        super().__init__("Requirement Analyzer", temperature=0.2)
    
    def get_system_prompt(self) -> str:
        return """You are an Expert Requirement Analyzer.

WORK AUTONOMOUSLY:
1. Parse the user requirements carefully
2. IDENTIFY hidden constraints (performance, scale, budget, compliance)
3. RECOMMEND 2 best architecture patterns WITHOUT waiting
4. SUGGEST the appropriate cloud provider
5. IDENTIFY potential risks PROACTIVELY

IMPORTANT: Make intelligent decisions yourself. Don't just list options.

Return JSON:
{
  "parsed_requirements": {"scale": "value", "latency": "value", ...},
  "identified_constraints": ["constraint1", "constraint2"],
  "recommended_patterns": ["pattern_id_1", "pattern_id_2"],
  "suggested_provider": "AWS|Azure|GCP",
  "identified_risks": ["risk1", "risk2"],
  "reasoning": "Your analysis process",
  "confidence": 0.85
}"""
    
    async def analyze(self, requirements: str) -> AgentResult:
        """Analyze requirements AUTONOMOUSLY."""
        
        logger.info(f"[{self.name}] Starting autonomous analysis...")
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=f"Analyze these requirements:\n\n{requirements}")
        ]
        
        response = self.client.invoke(messages)
        response_text = response.content
        
        try:
            result_json = json.loads(response_text)
        except json.JSONDecodeError:
            result_json = {"error": "Parse failed", "raw": response_text[:200]}
        
        logger.info(f"[{self.name}] Autonomously identified: {len(result_json.get('identified_risks', []))} risks")
        
        return AgentResult(
            agent_name=self.name,
            findings=result_json,
            reasoning=result_json.get("reasoning", ""),
            confidence=float(result_json.get("confidence", 0.7))
        )


class ArchitectureDesignerAgent(BaseAgent):
    """
    AUTONOMOUS Agent 2: Designs architecture based on requirements.
    
    Shows AUTONOMY by:
    - Making design decisions (not just listing)
    - Optimizing for constraints
    - Explaining choices
    - Working independently
    """
    
    def __init__(self):
        super().__init__("Architecture Designer", temperature=0.3)
    
    def get_system_prompt(self) -> str:
        return """You are a Principal Architect.

WORK AUTONOMOUSLY:
1. Design a COMPLETE system architecture
2. SELECT components (don't list options - CHOOSE)
3. EXPLAIN each design decision
4. OPTIMIZE for identified requirements
5. CREATE connections between components

AUTONOMOUS DECISION MAKING:
- Don't ask for clarification, make intelligent choices
- Recommend proven patterns
- Consider performance, cost, complexity

Return JSON:
{
  "architectural_style": "style name",
  "key_decisions": ["decision1 with reasoning", "decision2"],
  "components": [
    {
      "id": "component_id",
      "name": "Component Name",
      "type": "service|db|cache|gateway",
      "rationale": "Why this component"
    }
  ],
  "connections": [
    {
      "from": "component_id",
      "to": "component_id",
      "protocol": "HTTP|gRPC|SQL|Message"
    }
  ],
  "reasoning": "Overall architectural approach",
  "confidence": 0.88
}"""
    
    async def analyze(self, context: Dict[str, Any]) -> AgentResult:
        """Design architecture AUTONOMOUSLY."""
        
        logger.info(f"[{self.name}] Starting autonomous design...")
        
        requirements = context.get("requirements", "")
        req_analysis = context.get("requirement_analysis", {})
        
        prompt = f"""Based on these requirements and analysis:

REQUIREMENTS:
{requirements}

ANALYSIS FINDINGS:
{json.dumps(req_analysis, indent=2)}

Now AUTONOMOUSLY design the OPTIMAL architecture."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = self.client.invoke(messages)
        result_text = response.content
        
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json = {"error": "Parse failed"}
        
        logger.info(f"[{self.name}] Autonomously designed {len(result_json.get('components', []))} components")
        
        return AgentResult(
            agent_name=self.name,
            findings=result_json,
            reasoning=result_json.get("reasoning", ""),
            confidence=float(result_json.get("confidence", 0.8))
        )


class SecurityReviewerAgent(BaseAgent):
    """
    AUTONOMOUS Agent 3: Reviews security independently.
    
    Shows AUTONOMY by:
    - Analyzing threats proactively
    - Recommending solutions (not just identifying)
    - Working without prompting
    - Making security decisions
    """
    
    def __init__(self):
        super().__init__("Security Reviewer", temperature=0.1)  # Low temp = more deterministic
    
    def get_system_prompt(self) -> str:
        return """You are a Security Expert and Threat Analyst.

WORK AUTONOMOUSLY:
1. IDENTIFY all security threats in the architecture
2. RECOMMEND SPECIFIC security measures (don't just list issues)
3. PRESCRIBE encryption, auth, WAF strategies
4. VALIDATE compliance readiness
5. PRIORITIZE threats by severity

AUTONOMOUS DECISION MAKING:
- Make security recommendations yourself
- Suggest industry best practices
- Think about data sensitivity

Return JSON:
{
  "threats": [
    {
      "threat": "threat description",
      "severity": "Critical|High|Medium",
      "component": "affected_component",
      "mitigation": "specific solution"
    }
  ],
  "security_recommendations": ["recommendation1", "recommendation2"],
  "critical_gaps": ["gap1"],
  "reasoning": "Security analysis rationale",
  "confidence": 0.9
}"""
    
    async def analyze(self, context: Dict[str, Any]) -> AgentResult:
        """Review security AUTONOMOUSLY."""
        
        logger.info(f"[{self.name}] Starting autonomous security review...")
        
        architecture = context.get("architecture", {})
        requirements = context.get("requirements", "")
        
        prompt = f"""Review the security posture:

REQUIREMENTS:
{requirements}

ARCHITECTURE:
{json.dumps(architecture, indent=2)[:1000]}

AUTONOMOUSLY identify threats and prescribe specific security measures."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = self.client.invoke(messages)
        result_text = response.content
        
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json = {"error": "Parse failed"}
        
        logger.info(f"[{self.name}] Identified {len(result_json.get('threats', []))} threats autonomously")
        
        return AgentResult(
            agent_name=self.name,
            findings=result_json,
            reasoning=result_json.get("reasoning", ""),
            confidence=float(result_json.get("confidence", 0.85))
        )


class RecommendationEngineAgent(BaseAgent):
    """
    AUTONOMOUS Agent 4: Consolidates all findings.
    
    Shows AUTONOMY by:
    - Aggregating data from multiple agents
    - Making final decisions
    - Resolving conflicts intelligently
    - Creating executive summary
    """
    
    def __init__(self):
        super().__init__("Recommendation Engine", temperature=0.4)
    
    def get_system_prompt(self) -> str:
        return """You are a Final Decision Engine that consolidates multi-agent findings.

WORK AUTONOMOUSLY:
1. Aggregate findings from all agents
2. IDENTIFY any conflicts and resolve them intelligently
3. CREATE an executive summary
4. PRIORITIZE recommendations
5. MAKE final recommendations (don't just list options)

AUTONOMOUS DECISION MAKING:
- Decide what's most important
- Resolve agent disagreements
- Create actionable recommendations

Return JSON:
{
  "executive_summary": "High-level summary",
  "top_priorities": ["priority1", "priority2", "priority3"],
  "implementation_recommendations": ["rec1", "rec2"],
  "risk_mitigation": ["mitigation1"],
  "reasoning": "Decision-making process",
  "confidence": 0.85
}"""
    
    async def analyze(self, context: Dict[str, Any]) -> AgentResult:
        """Consolidate findings AUTONOMOUSLY."""
        
        logger.info(f"[{self.name}] Starting autonomous consolidation...")
        
        all_findings = json.dumps(context.get("all_agent_findings", {}), indent=2)
        
        prompt = f"""Consolidate these multi-agent findings autonomously:

{all_findings}

AUTONOMOUSLY create a consolidated recommendation."""
        
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=prompt)
        ]
        
        response = self.client.invoke(messages)
        result_text = response.content
        
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            result_json = {"error": "Parse failed"}
        
        logger.info(f"[{self.name}] Autonomously created consolidated report")
        
        return AgentResult(
            agent_name=self.name,
            findings=result_json,
            reasoning=result_json.get("reasoning", ""),
            confidence=float(result_json.get("confidence", 0.8))
        )


# =========================== #
# MULTI-AGENT ORCHESTRATOR   #
# =========================== #

class MultiAgentOrchestrator:
    """
    Orchestrates 4 autonomous agents.
    
    Shows MULTI-AGENT architecture by:
    - Parallel agent execution
    - Each agent makes independent decisions
    - Results aggregation
    - Comprehensive reporting
    """
    
    def __init__(self):
        self.agents = {
            "requirement_analyzer": RequirementAnalyzerAgent(),
            "architect": ArchitectureDesignerAgent(),
            "security_reviewer": SecurityReviewerAgent(),
            "recommendation_engine": RecommendationEngineAgent(),
        }
        self.agent_logs = []

    def get_architecture_for_diagram(self) -> Dict[str, Any]:
        """Extract architecture for diagram generation."""
        return {
            "components": self._extract_components(),
            "connections": self._extract_connections()
        }

    def _extract_components(self) -> List[Dict]:
        """Extract components from latest architecture."""
        if hasattr(self, '_last_architecture'):
            return self._last_architecture.get("components", [])
        return []

    def _extract_connections(self) -> List[Dict]:
        """Extract connections from latest architecture."""
        if hasattr(self, '_last_architecture'):
            return self._last_architecture.get("connections", [])
        return []
    
    async def orchestrate_autonomously(self, requirements: str, conversation_id: str = "default") -> Dict[str, Any]:
        """
        AUTONOMOUSLY orchestrate all agents.
        
        Flow:
        1. Requirement Analyzer → autonomous analysis
        2. Architect → autonomous design
        3. Security Reviewer → autonomous review
        4. Recommendation Engine → autonomous consolidation
        
        Each agent makes independent decisions WITHOUT user guidance.
        """
        
        logger.info(f"\n{'='*60}")
        logger.info(f"[{conversation_id}] MULTI-AGENT AUTONOMOUS ORCHESTRATION STARTING")
        logger.info(f"{'='*60}\n")
        
        self.agent_logs = []
        
        try:
            # PHASE 1: REQUIREMENT ANALYSIS (Autonomous)
            logger.info("PHASE 1: Requirement Analyzer Agent (AUTONOMOUS)")
            logger.info("-" * 60)
            
            req_result = await self.agents["requirement_analyzer"].analyze(requirements)
            
            log_entry = {
                "phase": 1,
                "agent": req_result.agent_name,
                "status": "✅ AUTONOMOUS ANALYSIS COMPLETE",
                "findings_count": len(req_result.findings),
                "confidence": req_result.confidence,
                "reasoning": req_result.reasoning[:200] + "..." if len(req_result.reasoning) > 200 else req_result.reasoning,
                "timestamp": req_result.timestamp
            }
            self.agent_logs.append(log_entry)
            logger.info(f"✅ {req_result.agent_name}: {len(req_result.findings)} findings, confidence: {req_result.confidence}")
            
            # PHASE 2: ARCHITECTURE DESIGN (Autonomous)
            logger.info("\nPHASE 2: Architecture Designer Agent (AUTONOMOUS)")
            logger.info("-" * 60)
            
            architect_result = await self.agents["architect"].analyze({
                "requirements": requirements,
                "requirement_analysis": req_result.findings
            })

            self._last_architecture = architect_result.findings
            
            components_count = len(architect_result.findings.get("components", []))
            log_entry = {
                "phase": 2,
                "agent": architect_result.agent_name,
                "status": "✅ AUTONOMOUS DESIGN COMPLETE",
                "components_designed": components_count,
                "confidence": architect_result.confidence,
                "reasoning": architect_result.reasoning[:200] + "..." if len(architect_result.reasoning) > 200 else architect_result.reasoning,
                "timestamp": architect_result.timestamp
            }
            self.agent_logs.append(log_entry)
            logger.info(f"✅ {architect_result.agent_name}: Designed {components_count} components, confidence: {architect_result.confidence}")
            
            # PHASE 3: SECURITY REVIEW (Autonomous - NO USER NEEDED)
            logger.info("\nPHASE 3: Security Reviewer Agent (AUTONOMOUS - NO PROMPT NEEDED)")
            logger.info("-" * 60)
            
            security_result = await self.agents["security_reviewer"].analyze({
                "architecture": architect_result.findings,
                "requirements": requirements
            })
            
            threats_count = len(security_result.findings.get("threats", []))
            log_entry = {
                "phase": 3,
                "agent": security_result.agent_name,
                "status": "✅ AUTONOMOUS SECURITY REVIEW COMPLETE",
                "threats_identified": threats_count,
                "confidence": security_result.confidence,
                "reasoning": security_result.reasoning[:200] + "..." if len(security_result.reasoning) > 200 else security_result.reasoning,
                "timestamp": security_result.timestamp
            }
            self.agent_logs.append(log_entry)
            logger.info(f"✅ {security_result.agent_name}: AUTONOMOUSLY identified {threats_count} threats, confidence: {security_result.confidence}")
            
            # PHASE 4: RECOMMENDATION CONSOLIDATION (Autonomous)
            logger.info("\nPHASE 4: Recommendation Engine Agent (AUTONOMOUS CONSOLIDATION)")
            logger.info("-" * 60)
            
            rec_result = await self.agents["recommendation_engine"].analyze({
                "all_agent_findings": {
                    "requirement_analysis": req_result.findings,
                    "architecture": architect_result.findings,
                    "security_review": security_result.findings
                }
            })
            
            log_entry = {
                "phase": 4,
                "agent": rec_result.agent_name,
                "status": "✅ AUTONOMOUS CONSOLIDATION COMPLETE",
                "recommendations": len(rec_result.findings.get("implementation_recommendations", [])),
                "confidence": rec_result.confidence,
                "reasoning": rec_result.reasoning[:200] + "..." if len(rec_result.reasoning) > 200 else rec_result.reasoning,
                "timestamp": rec_result.timestamp
            }
            self.agent_logs.append(log_entry)
            logger.info(f"✅ {rec_result.agent_name}: AUTONOMOUSLY consolidated findings, confidence: {rec_result.confidence}")
            
            logger.info(f"\n{'='*60}")
            logger.info("🎯 MULTI-AGENT AUTONOMOUS ORCHESTRATION COMPLETE")
            logger.info(f"{'='*60}\n")
            
            # CONSOLIDATE FINAL RESULT
            final_result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "conversation_id": conversation_id,
                
                # AUTONOMOUS ANALYSIS RESULTS
                "requirement_analysis": req_result.findings,
                "architecture_design": architect_result.findings,
                "security_review": security_result.findings,
                "recommendations": rec_result.findings,
                
                # AGENT REASONING LOGS (shows autonomy)
                "agent_execution_log": self.agent_logs,
                
                # SHOW THAT AGENTS WORKED INDEPENDENTLY
                "agents_involved": [
                    req_result.agent_name,
                    architect_result.agent_name,
                    security_result.agent_name,
                    rec_result.agent_name
                ],

                "diagram_data": {
                    "components": architect_result.findings.get("components", []),
                    "connections": architect_result.findings.get("connections", [])
                },
                
                "total_confidence": sum([
                    req_result.confidence,
                    architect_result.confidence,
                    security_result.confidence,
                    rec_result.confidence
                ]) / 4
            }
            
            return final_result
        
        except Exception as e:
            logger.error(f"[{conversation_id}] Orchestration error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent_logs": self.agent_logs
            }


# Singleton
_orchestrator = None

def get_orchestrator():
    """Get or create orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator


# For testing
if __name__ == "__main__":
    async def test():
        orchestrator = get_orchestrator()
        result = await orchestrator.orchestrate_autonomously(
            "I need an e-commerce platform for 50K users with payment processing"
        )
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())





# """
# architecture_agent.py

# Core logic for the Architecture Design Assistant.

# - Uses Azure OpenAI (via LangChain's ChatOpenAI).
# - Wrapped in a LangGraph workflow with an in-memory checkpointer
#   (MemorySaver) so the backend keeps per-conversation state.
# - Now explicitly REFINES the previous architecture plan (if any)
#   instead of redesigning from scratch on follow-up prompts.
# """

# import json
# from typing import Dict, Any, List, TypedDict, Annotated
# import httpx
# from langchain_openai import ChatOpenAI
# import config
# import logging
# import traceback
# from openai import InternalServerError

# import operator
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)  # later you can change to DEBUG

# # ===========================
# # AZURE OPENAI CLIENT
# # ===========================

# # Local-only POC workaround to avoid corporate SSL cert issues
# http_client = httpx.Client(verify=False)

# client = ChatOpenAI(
#     api_key=config.AZURE_OPENAI_API_KEY,
#     base_url=config.AZURE_OPENAI_ENDPOINT,
#     model=config.OPENAI_MODEL,
#     http_client=http_client,
#     # Lower temperature to improve determinism / consistency
#     temperature=0.1,
# )

# # ===========================
# # Load templates
# # ===========================

# with open(config.TEMPLATES_PATH, "r", encoding="utf-8") as f:
#     TEMPLATE_DATA = json.load(f)


# def build_prompt_messages(
#     user_message: str,
#     previous_arch_plan: Dict[str, Any] | None,
# ) -> List[Dict[str, Any]]:
#     """
#     Build the message list for the LLM.

#     user_message:
#         The full accumulated requirements text
#         (first prompt + any refinements appended by the backend).

#     previous_arch_plan:
#         The last architecture plan produced for this conversation
#         (if any). When present, the model is told to REFINE it,
#         not redesign from scratch.
#     """
#     template_summaries = [
#         {
#             "id": p["id"],
#             "name": p["name"],
#             "description": p["description"],
#         }
#         for p in TEMPLATE_DATA.get("patterns", [])
#     ]

#     templates_str = json.dumps(template_summaries, indent=2)

#     # ---- SYSTEM PROMPT ----
#     system_content = (
#         "You are an Architecture Design Assistant for IT systems. "
#         "Your job is to take high-level requirements and propose a system architecture.\n\n"
#         "You have access to a small library of architecture patterns. "
#         "Each pattern has an id, name, and description. Use them as reusable reference designs.\n\n"
#         "Return ONLY JSON (no markdown outside the JSON block, no extra text). "
#         "The JSON MUST have this structure:\n"
#         "{\n"
#         "  \"summary\": \"An HTML-formatted architecture summary.\",\n"
#         "  \"pattern_id\": \"id of the pattern you are closest to (or 'custom' if none fits)\",\n"
#         "  \"components\": [\n"
#         "    {\"id\": \"short_id\", \"label\": \"Readable name\", \"type\": \"e.g. web, app, db, cache, queue, mobile_client\"}\n"
#         "  ],\n"
#         "  \"connections\": [\n"
#         "    {\"from\": \"component_id\", \"to\": \"component_id\", \"label\": \"protocol or purpose\"}\n"
#         "  ]\n"
#         "}\n"
#         "IDs must be valid Graphviz node identifiers (letters, digits, underscores only). "
#         "Use about 4–12 components to keep the diagram readable.\n\n"
#         "IMPORTANT: The `summary` field MUST be valid HTML, not markdown. Use tags like:\n"
#         "- <h3>Overview</h3>\n"
#         "- <h3>Key Components</h3>\n"
#         "- <h3>Data Flow</h3>\n"
#         "- <h3>Scalability & Reliability</h3>\n"
#         "Within each section, use <ul><li>...</li></ul> bullet lists.\n\n"
#         "SUMMARY LENGTH RULES:\n"
#         "- Keep the HTML formatting EXACTLY the same (h3 headings + bullet lists).\n"
#         "- Keep all <ul><li>...</li></ul> bullet lists.\n"
#         "- Make the summary concise: shorten each bullet point using brief, telegraphic text.\n"
#         "- Keep the meaning but remove verbosity.\n"
#         "- Target 40–60% of the usual summary length.\n\n"
#         "REFINEMENT RULES:\n"
#         "- If a previous architecture plan is provided, treat it as the BASELINE.\n"
#         "- You MUST keep existing component IDs and labels as stable as possible.\n"
#         "- Prefer to ADD components or connections rather than renaming or deleting.\n"
#         "- Only change or remove existing components if the new requirements clearly conflict.\n"
#         "- If a previous pattern_id is provided, keep the same pattern_id unless the user explicitly asks to change the pattern.\n"
#     )

#     # ---- USER PROMPT ----
#     user_content_parts: List[str] = []

#     user_content_parts.append("Here are the available architecture patterns:\n")
#     user_content_parts.append(templates_str)

#     if previous_arch_plan:
#         # We are in a follow-up turn; show the previous plan to the model
#         user_content_parts.append(
#             "\n\nHere is the PREVIOUS architecture plan JSON. "
#             "This is your baseline. REFINE this plan instead of redesigning from scratch:\n"
#         )
#         user_content_parts.append(json.dumps(previous_arch_plan, indent=2))

#         user_content_parts.append(
#             "\n\nThe user has provided NEW requirements / refinements. "
#             "Update the existing architecture minimally to satisfy them:\n"
#         )
#         user_content_parts.append(user_message)
#     else:
#         # First turn: design from scratch based on full requirements text
#         user_content_parts.append(
#             "\n\nThe FULL set of user requirements (including any refinements) is:\n"
#         )
#         user_content_parts.append(user_message)

#     user_content = "".join(user_content_parts)

#     messages = [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": user_content},
#     ]
#     return messages


# def _call_model(
#     user_message: str,
#     previous_arch_plan: Dict[str, Any] | None,
# ) -> Dict[str, Any]:
#     """
#     Internal helper that calls the LLM and parses the JSON architecture plan.

#     user_message:
#         The full accumulated requirements text built by the backend.

#     previous_arch_plan:
#         The last architecture plan for this conversation (if any).
#         When present, the model is instructed to REFINE it.
#     """
#     if not config.AZURE_OPENAI_API_KEY:
#         raise RuntimeError("Missing Azure OpenAI API key in config.py")

#     messages = build_prompt_messages(user_message, previous_arch_plan)

#     print("=== FULL REQUIREMENTS SENT TO MODEL ===")
#     print(user_message)
#     print("=== PREVIOUS ARCH PLAN (if any) ===")
#     print(json.dumps(previous_arch_plan, indent=2) if previous_arch_plan else "None")
#     print("=======================================")

#     # Build a single prompt string for ChatOpenAI (as in your original version)
#     system_content = messages[0]["content"]
#     user_content = messages[1]["content"]
#     full_prompt = system_content + "\n\n" + user_content

#     try:
#         # ChatOpenAI interface: use invoke()
#         llm_result = client.invoke(full_prompt)

#         # llm_result is a ChatMessage-like object; get the text content
#         raw_text = getattr(llm_result, "content", str(llm_result))

#         print("=== RAW MODEL OUTPUT ===")
#         print(raw_text)
#         print("========================")

#         # ---- CLEANUP LOGIC ----
#         import re

#         clean_text = raw_text.strip()

#         # If the model wrapped JSON in ```json ... ```
#         if clean_text.startswith("```"):
#             # Extract the first {...} block from inside
#             match = re.search(r"\{[\s\S]*\}", clean_text)
#             if match:
#                 clean_text = match.group(0)

#         print("=== CLEAN JSON CANDIDATE ===")
#         print(clean_text)
#         print("============================")

#         try:
#             arch_plan = json.loads(clean_text)  # ✅ use cleaned text
#         except Exception as e:
#             print("JSON parse failed, using fallback architecture.")
#             print("JSON error:", e)
#             arch_plan = _fallback_architecture("Could not parse JSON from model output.")

#     except InternalServerError as e:
#         # This is a 5xx from your gateway (genailab.tcs.in)
#         logger.error("Azure gateway returned 500. Status: %s", e.status_code)
#         try:
#             logger.error("Response body: %s", e.response.text)
#         except Exception:
#             pass
#         raise RuntimeError(
#             "Server error from genailab.tcs.in (500). Check gateway logs / configuration."
#         ) from e

#     except Exception as ex:
#         # 🔍 Detailed logging
#         logger.error("Azure OpenAI call failed: %s", ex)
#         logger.error("Exception type: %s", type(ex).__name__)
#         logger.error("Full traceback:\n%s", traceback.format_exc())
#         raise RuntimeError(
#             "Connection error: unable to reach Azure OpenAI. Please check network / VPN."
#         ) from ex

#     # Ensure keys exist
#     arch_plan.setdefault("summary", "No summary provided.")
#     arch_plan.setdefault("pattern_id", "unknown")
#     arch_plan.setdefault("components", [])
#     arch_plan.setdefault("connections", [])

#     print("=== PARSED ARCH PLAN ===")
#     print(json.dumps(arch_plan, indent=2))
#     print("========================")

#     return arch_plan


# def _fallback_architecture(reason: str) -> Dict[str, Any]:
#     """
#     Fallback architecture plan when something goes wrong.
#     """
#     return {
#         "summary": (
#             f"Fallback architecture used because: {reason}\n\n"
#             "This is a simple three-tier web application."
#         ),
#         "pattern_id": "fallback_three_tier",
#         "components": [
#             {"id": "client", "label": "Client", "type": "client"},
#             {"id": "web", "label": "Web Server", "type": "web"},
#             {"id": "app", "label": "Application Server", "type": "app"},
#             {"id": "db", "label": "Database", "type": "database"},
#         ],
#         "connections": [
#             {"from": "client", "to": "web", "label": "HTTP/HTTPS"},
#             {"from": "web", "to": "app", "label": "Internal HTTP"},
#             {"from": "app", "to": "db", "label": "SQL"},
#         ],
#     }


# # ===========================
# # LangGraph: state + memory
# # ===========================

# class ArchState(TypedDict):
#     """
#     State for the LangGraph workflow.

#     - messages: list of requirement text snapshots
#       (we append each call's full requirements string here so we
#        always know the latest).
#     - arch_plan: the latest parsed architecture JSON from the model.
#     - arch_history: list of ALL architecture plans produced so far
#       for this conversation (used to get the previous plan on follow-ups).
#     """
#     messages: Annotated[List[str], operator.add]
#     arch_plan: Dict[str, Any]
#     arch_history: Annotated[List[Dict[str, Any]], operator.add]


# def _llm_node(state: ArchState) -> ArchState:
#     """
#     LangGraph node that calls the model using the latest requirements text.

#     The MemorySaver checkpointer keeps the 'messages' and 'arch_history'
#     per thread_id, so follow-up turns can refine the previous architecture.
#     """
#     messages = state.get("messages") or []
#     if not messages:
#         raise RuntimeError("No requirements text provided to LLM node.")

#     # Latest full requirements text (first prompt + refinements)
#     latest_requirements = messages[-1]

#     arch_history = state.get("arch_history") or []
#     previous_arch_plan = arch_history[-1] if arch_history else None

#     arch_plan = _call_model(latest_requirements, previous_arch_plan)

#     # Return only the NEW plan as a delta for arch_history.
#     # MemorySaver + operator.add will append it to the stored list.
#     return {
#         "messages": [],             # we've consumed messages for this step
#         "arch_plan": arch_plan,     # latest plan
#         "arch_history": [arch_plan] # append-only history
#     }


# # Build the LangGraph workflow with in-memory checkpointing
# _graph_builder = StateGraph(ArchState)
# _graph_builder.add_node("llm", _llm_node)
# _graph_builder.set_entry_point("llm")
# _graph_builder.add_edge("llm", END)

# _checkpointer = MemorySaver()
# _arch_graph = _graph_builder.compile(checkpointer=_checkpointer)


# def call_llm_for_architecture(
#     user_message: str,
#     thread_id: str = "default",
# ) -> Dict[str, Any]:
#     """
#     Public entry point used by the Flask app.

#     user_message:
#         The full requirements text (first prompt + refinements), as built
#         by app.py for backward-compatible behavior.

#     thread_id:
#         Per-conversation identifier used by LangGraph's MemorySaver
#         to maintain server-side state across turns. Should come from
#         the frontend's conversation_id.
#     """
#     if not config.AZURE_OPENAI_API_KEY:
#         raise RuntimeError("Missing Azure OpenAI API key in config.py")

#     initial_state: ArchState = {
#         "messages": [user_message],
#         "arch_plan": {},
#         "arch_history": [],
#     }

#     final_state = _arch_graph.invoke(
#         initial_state,
#         config={"configurable": {"thread_id": thread_id}},
#     )

#     arch_plan = final_state.get("arch_plan") or _fallback_architecture(
#         "Missing arch_plan from LangGraph state."
#     )
#     return arch_plan

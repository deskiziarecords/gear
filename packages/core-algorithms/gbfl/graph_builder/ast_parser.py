# packages/core-algorithms/gbfl/graph_builder/ast_parser.py

from tree_sitter import Language, Parser, Node, Query
import tree_sitter_python as tspython
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
import networkx as nx

@dataclass
class MethodNode:
    id: str  # file_path::class_name::method_name or file_path::method_name
    file_path: str
    class_name: Optional[str]
    method_name: str
    start_line: int
    end_line: int
    source_code: str
    
@dataclass
class CallSite:
    caller: str  # MethodNode.id
    callee: str  # MethodNode.id or external method name
    line_number: int
    is_external: bool  # True if calling library/undefined method

class TreeSitterGraphBuilder:
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(tspython.language())
        
        # Pre-compile queries for performance
        self._init_queries()
    
    def _init_queries(self):
        """Initialize Tree-sitter queries for pattern matching [^25^][^28^]"""
        
        # Query to find function/method definitions
        self.func_def_query = Query(tspython.language(), '''
        (function_definition
            name: (identifier) @func.name
            parameters: (parameters) @func.params
            body: (block) @func.body) @func.def
            
        (class_definition
            name: (identifier) @class.name
            body: (block
                (function_definition
                    name: (identifier) @method.name
                    parameters: (parameters) @method.params
                    body: (block) @method.body) @method.def)) @class.def
        ''')
        
        # Query to find function calls
        self.call_query = Query(tspython.language(), '''
        (call
            function: [
                (identifier) @call.name
                (attribute
                    object: (_)? @call.obj
                    attribute: (identifier) @call.attr)
                ]) @call.site
        ''')
    
    def build_graph(self, repo_path: str) -> nx.DiGraph:
        """
        Build method-level dependency graph combining AST and call relationships.
        Returns a NetworkX DiGraph where nodes are methods and edges are calls.
        """
        graph = nx.DiGraph()
        method_map: Dict[str, MethodNode] = {}
        
        # Phase 1: Discover all methods
        for py_file in Path(repo_path).rglob("*.py"):
            methods = self._extract_methods(py_file)
            for method in methods:
                method_map[method.id] = method
                graph.add_node(method.id, 
                              data=method,
                              node_type='method',
                              file=method.file_path)
        
        # Phase 2: Build call graph
        for py_file in Path(repo_path).rglob("*.py"):
            calls = self._extract_calls(py_file, method_map)
            for call in calls:
                if call.is_external:
                    # Add external node if not exists
                    if not graph.has_node(call.callee):
                        graph.add_node(call.callee, 
                                     node_type='external',
                                     file='external')
                
                # Add edge with call site metadata
                graph.add_edge(call.caller, call.callee,
                             line_number=call.line_number,
                             edge_type='calls')
        
        # Phase 3: Add structural edges (parent-child in AST)
        self._add_structural_edges(graph, method_map)
        
        return graph
    
    def _extract_methods(self, file_path: Path) -> List[MethodNode]:
        """Extract all method definitions from a Python file"""
        methods = []
        source = file_path.read_text(encoding='utf-8')
        tree = self.parser.parse(bytes(source, "utf8"))
        
        # Capture function definitions
        captures = self.func_def_query.captures(tree.root_node)
        
        current_class = None
        for node, capture_name in captures:
            if capture_name == 'class.name':
                current_class = source[node.start_byte:node.end_byte]
            elif capture_name in ('func.name', 'method.name'):
                method_name = source[node.start_byte:node.end_byte]
                class_name = current_class if capture_name == 'method.name' else None
                
                # Find parent function_definition node to get line numbers
                func_node = node.parent
                while func_node and func_node.type != 'function_definition':
                    func_node = func_node.parent
                
                if func_node:
                    method_id = self._create_method_id(str(file_path), class_name, method_name)
                    methods.append(MethodNode(
                        id=method_id,
                        file_path=str(file_path),
                        class_name=class_name,
                        method_name=method_name,
                        start_line=func_node.start_point[0] + 1,
                        end_line=func_node.end_point[0] + 1,
                        source_code=source[func_node.start_byte:func_node.end_byte]
                    ))
        
        return methods
    
    def _extract_calls(self, file_path: Path, method_map: Dict[str, MethodNode]) -> List[CallSite]:
        """Extract all call sites from a Python file"""
        calls = []
        source = file_path.read_text(encoding='utf-8')
        tree = self.parser.parse(bytes(source, "utf8"))
        
        # First, map line numbers to containing methods
        line_to_method: Dict[int, str] = {}
        for method in method_map.values():
            if method.file_path == str(file_path):
                for line in range(method.start_line, method.end_line + 1):
                    line_to_method[line] = method.id
        
        # Find all calls
        captures = self.call_query.captures(tree.root_node)
        
        for node, capture_name in captures:
            if capture_name == 'call.site':
                line_num = node.start_point[0] + 1
                caller_id = line_to_method.get(line_num)
                
                if not caller_id:
                    continue
                
                # Determine callee name
                call_text = source[node.start_byte:node.end_byte]
                
                # Simple heuristic: resolve call target
                callee_name = self._resolve_call_target(node, source, method_map)
                is_external = callee_name not in method_map
                
                calls.append(CallSite(
                    caller=caller_id,
                    callee=callee_name,
                    line_number=line_num,
                    is_external=is_external
                ))
        
        return calls
    
    def _resolve_call_target(self, call_node: Node, source: str, 
                            method_map: Dict[str, MethodNode]) -> str:
        """Resolve a call node to a method ID or external name"""
        # Extract function name from call
        func_node = None
        for child in call_node.children:
            if child.type == 'identifier':
                func_node = child
                break
            elif child.type == 'attribute':
                # method call: obj.method()
                attr_node = child.child_by_field_name('attribute')
                if attr_node:
                    func_node = attr_node
                break
        
        if not func_node:
            return "unknown"
        
        func_name = source[func_node.start_byte:func_node.end_byte]
        
        # Try to find matching method in map (simplified resolution)
        # In production, you'd use import resolution and type inference
        for method_id in method_map:
            if method_id.endswith(f"::{func_name}"):
                return method_id
        
        return func_name  # Return as external reference
    
    def _create_method_id(self, file_path: str, class_name: Optional[str], 
                         method_name: str) -> str:
        """Create unique identifier for a method"""
        if class_name:
            return f"{file_path}::{class_name}::{method_name}"
        return f"{file_path}::{method_name}"
    
    def _add_structural_edges(self, graph: nx.DiGraph, 
                             method_map: Dict[str, MethodNode]):
        """Add CONTAINS edges for nested methods and class membership"""
        # Group methods by file and class
        file_methods: Dict[str, List[MethodNode]] = {}
        for method in method_map.values():
            file_methods.setdefault(method.file_path, []).append(method)
        
        for file_path, methods in file_methods.items():
            # Sort by line range to detect nesting
            methods.sort(key=lambda m: (m.start_line, -m.end_line))
            
            # Simple nesting detection (can be enhanced with proper AST parent tracking)
            for i, outer in enumerate(methods):
                for inner in methods[i+1:]:
                    if inner.start_line >= outer.start_line and \
                       inner.end_line <= outer.end_line:
                        graph.add_edge(outer.id, inner.id, edge_type='contains')
                        break

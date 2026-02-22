# auto_repair/config.py

"""
Configuration management for auto_repair module.

Supports multiple sources (env vars, files, code) with validation and defaults.
Uses Pydantic for type safety and runtime validation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
from enum import Enum
from dataclasses import dataclass, field
import json
import yaml

# Try to import Pydantic, fall back to dataclasses if not available
try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic.types import DirectoryPath, FilePath
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: Pydantic not installed. Using dataclasses with basic validation.")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class SBFLFormula(str, Enum):
    OCHIAI = "ochiai"
    TARANTULA = "tarantula"
    OP2 = "op2"
    BARINEL = "barinel"
    DSTAR = "dstar"
    JACCARD = "jaccard"

class GNNArchitecture(str, Enum):
    GRAPHSAGE = "graphsage"
    GAT = "gat"
    GCN = "gcn"
    DEPGRAPH = "depgraph"

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"

class SandboxType(str, Enum):
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    FIRECRACKER = "firecracker"
    NAMESPACE = "namespace"  # Linux namespaces only

class DatabaseType(str, Enum):
    POSTGRES = "postgresql"
    SQLITE = "sqlite"
    MONGODB = "mongodb"

class CacheType(str, Enum):
    REDIS = "redis"
    MEMCACHED = "memcached"
    IN_MEMORY = "in_memory"


# =============================================================================
# COMPONENT CONFIGURATIONS
# =============================================================================

if PYDANTIC_AVAILABLE:
    
    class GBFLConfig(BaseModel):
        """Graph-Based Fault Localization configuration."""
        
        formula: SBFLFormula = Field(
            default=SBFLFormula.OCHIAI,
            description="SBFL formula for initial ranking"
        )
        use_gnn: bool = Field(
            default=True,
            description="Enable GNN refinement layer"
        )
        gnn_architecture: GNNArchitecture = Field(
            default=GNNArchitecture.GRAPHSAGE,
            description="GNN architecture for refinement"
        )
        gnn_model_path: Optional[Path] = Field(
            default=None,
            description="Path to pretrained GNN weights"
        )
        gnn_hidden_channels: int = Field(
            default=128,
            ge=16,
            le=1024,
            description="Hidden layer dimension for GNN"
        )
        gnn_num_layers: int = Field(
            default=3,
            ge=1,
            le=10,
            description="Number of GNN layers"
        )
        gnn_dropout: float = Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description="Dropout rate for GNN"
        )
        method_level: bool = Field(
            default=True,
            description="Use method-level graphs (vs statement-level)"
        )
        max_graph_nodes: int = Field(
            default=10000,
            description="Maximum nodes in graph before sampling"
        )
        
        @validator('gnn_model_path')
        def validate_model_path(cls, v):
            if v is not None and not v.exists():
                raise ValueError(f"GNN model path does not exist: {v}")
            return v

    class REVRConfig(BaseModel):
        """Reverse Execution Verification & Repair configuration."""
        
        enabled: bool = Field(
            default=True,
            description="Enable REVR analysis"
        )
        enable_symexec: bool = Field(
            default=True,
            description="Enable symbolic execution"
        )
        enable_concolic: bool = Field(
            default=True,
            description="Enable concolic execution"
        )
        smt_solver: Literal["z3", "cvc5", "yices"] = Field(
            default="z3",
            description="SMT solver backend"
        )
        smt_timeout: int = Field(
            default=30,
            ge=1,
            le=300,
            description="SMT solver timeout in seconds"
        )
        max_path_length: int = Field(
            default=100,
            ge=10,
            le=1000,
            description="Maximum path length for symbolic execution"
        )
        max_loop_unroll: int = Field(
            default=5,
            ge=1,
            le=100,
            description="Maximum loop unrolling iterations"
        )
        abstract_domain: Literal["interval", "predicate", "heap", "combined"] = Field(
            default="combined",
            description="Abstract interpretation domain"
        )
        widening_threshold: int = Field(
            default=3,
            description="Iterations before widening in fixpoint"
        )
        reverse_depth: int = Field(
            default=50,
            description="Maximum depth for reverse execution"
        )
        
        # Domain-specific settings
        interval_precision: int = Field(
            default=64,
            description="Bit width for interval arithmetic"
        )
        predicate_set_size: int = Field(
            default=20,
            description="Maximum predicates for predicate abstraction"
        )

    class PFSConfig(BaseModel):
        """Probabilistic Fix Synthesis configuration."""
        
        enabled: bool = Field(
            default=True,
            description="Enable patch synthesis"
        )
        model_path: Optional[Path] = Field(
            default=None,
            description="Path to Prophet-style model weights"
        )
        max_patches: int = Field(
            default=100,
            ge=1,
            le=1000,
            description="Maximum patches to generate"
        )
        max_templates: int = Field(
            default=50,
            description="Maximum template instantiations"
        )
        enable_semantic_synthesis: bool = Field(
            default=True,
            description="Enable angelic execution synthesis"
        )
        semantic_timeout: int = Field(
            default=60,
            description="Timeout for semantic synthesis in seconds"
        )
        feature_dimensions: int = Field(
            default=50,
            description="Feature vector dimension"
        )
        template_temperature: float = Field(
            default=0.8,
            ge=0.0,
            le=2.0,
            description="Temperature for template selection"
        )
        rank_by: Literal["probability", "semantic", "combined"] = Field(
            default="combined",
            description="Ranking strategy"
        )

    class FUConfig(BaseModel):
        """Fix Understanding (LLM) configuration."""
        
        enabled: bool = Field(
            default=True,
            description="Enable LLM-based understanding"
        )
        provider: LLMProvider = Field(
            default=LLMProvider.OPENAI,
            description="LLM provider"
        )
        model: str = Field(
            default="gpt-4",
            description="Model name/deployment"
        )
        api_key: Optional[str] = Field(
            default=None,
            description="API key (or use env var)"
        )
        api_base: Optional[str] = Field(
            default=None,
            description="Custom API base URL"
        )
        temperature: float = Field(
            default=0.3,
            ge=0.0,
            le=2.0,
            description="LLM temperature"
        )
        max_tokens: int = Field(
            default=2000,
            description="Maximum tokens for responses"
        )
        request_timeout: int = Field(
            default=30,
            description="LLM request timeout"
        )
        cache_responses: bool = Field(
            default=True,
            description="Cache LLM responses"
        )
        cache_ttl: int = Field(
            default=3600,
            description="Cache TTL in seconds"
        )
        max_concurrent_requests: int = Field(
            default=10,
            description="Max concurrent LLM calls"
        )
        rate_limit_rpm: int = Field(
            default=60,
            description="Rate limit (requests per minute)"
        )

    class ValidationConfig(BaseModel):
        """Patch validation configuration."""
        
        sandbox_type: SandboxType = Field(
            default=SandboxType.DOCKER,
            description="Sandbox technology"
        )
        sandbox_image: str = Field(
            default="auto-repair-sandbox:latest",
            description="Docker image for sandbox"
        )
        sandbox_timeout: int = Field(
            default=300,
            description="Sandbox execution timeout"
        )
        max_memory_mb: int = Field(
            default=2048,
            description="Max memory per sandbox"
        )
        max_cpu_cores: float = Field(
            default=2.0,
            description="Max CPU cores per sandbox"
        )
        network_enabled: bool = Field(
            default=False,
            description="Enable network in sandbox"
        )
        required_test_coverage: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Required test coverage threshold"
        )
        differential_testing: bool = Field(
            default=True,
            description="Enable differential testing"
        )
        static_analysis_tools: List[str] = Field(
            default=["pylint", "mypy", "bandit"],
            description="Static analysis tools to run"
        )

    class OrchestrationConfig(BaseModel):
        """Workflow orchestration configuration."""
        
        max_repair_attempts: int = Field(
            default=3,
            description="Maximum repair iterations"
        )
        human_in_the_loop: bool = Field(
            default=True,
            description="Require human approval"
        )
        auto_apply_confidence_threshold: float = Field(
            default=0.95,
            ge=0.0,
            le=1.0,
            description="Auto-apply threshold"
        )
        rollback_on_failure: bool = Field(
            default=True,
            description="Enable automatic rollback"
        )
        parallel_localization: bool = Field(
            default=True,
            description="Run localizers in parallel"
        )
        parallel_synthesis: bool = Field(
            default=True,
            description="Generate patches in parallel"
        )
        state_persistence: DatabaseType = Field(
            default=DatabaseType.POSTGRES,
            description="State database type"
        )

    class KnowledgeConfig(BaseModel):
        """Knowledge base configuration."""
        
        models_dir: Path = Field(
            default=Path("./models"),
            description="Directory for model weights"
        )
        embeddings_dir: Path = Field(
            default=Path("./embeddings"),
            description="Directory for code embeddings"
        )
        historical_patches_dir: Path = Field(
            default=Path("./historical_patches"),
            description="Directory for training data"
        )
        patterns_file: Optional[Path] = Field(
            default=None,
            description="Common bug patterns database"
        )
        enable_online_learning: bool = Field(
            default=False,
            description="Enable continuous learning"
        )
        feedback_retention_days: int = Field(
            default=90,
            description="Days to retain feedback"
        )

    class TelemetryConfig(BaseModel):
        """Observability configuration."""
        
        log_level: LogLevel = Field(
            default=LogLevel.INFO,
            description="Logging level"
        )
        log_format: Literal["json", "text"] = Field(
            default="json",
            description="Log format"
        )
        metrics_enabled: bool = Field(
            default=True,
            description="Enable metrics collection"
        )
        metrics_backend: Literal["prometheus", "statsd", "cloudwatch"] = Field(
            default="prometheus",
            description="Metrics backend"
        )
        tracing_enabled: bool = Field(
            default=True,
            description="Enable distributed tracing"
        )
        jaeger_endpoint: Optional[str] = Field(
            default=None,
            description="Jaeger collector endpoint"
        )
        audit_log_path: Path = Field(
            default=Path("./audit.log"),
            description="Immutable audit log path"
        )

    class APIConfig(BaseModel):
        """API server configuration."""
        
        host: str = Field(
            default="0.0.0.0",
            description="Bind host"
        )
        port: int = Field(
            default=8080,
            ge=1,
            le=65535,
            description="Bind port"
        )
        workers: int = Field(
            default=4,
            description="Number of worker processes"
        )
        cors_origins: List[str] = Field(
            default=["*"],
            description="Allowed CORS origins"
        )
        auth_enabled: bool = Field(
            default=True,
            description="Enable authentication"
        )
        jwt_secret: Optional[str] = Field(
            default=None,
            description="JWT signing secret"
        )
        rate_limit_per_minute: int = Field(
            default=100,
            description="API rate limit"
        )
        max_request_size_mb: int = Field(
            default=10,
            description="Max request body size"
        )
        websocket_enabled: bool = Field(
            default=True,
            description="Enable WebSocket for streaming"
        )

    class AutoRepairConfig(BaseModel):
        """Root configuration for auto_repair module."""
        
        # Component configs
        gbfl: GBFLConfig = Field(default_factory=GBFLConfig)
        revr: REVRConfig = Field(default_factory=REVRConfig)
        pfs: PFSConfig = Field(default_factory=PFSConfig)
        fu: FUConfig = Field(default_factory=FUConfig)
        validation: ValidationConfig = Field(default_factory=ValidationConfig)
        orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
        knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
        telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
        api: APIConfig = Field(default_factory=APIConfig)
        
        # Global settings
        environment: Literal["development", "staging", "production"] = Field(
            default="development",
            description="Runtime environment"
        )
        debug: bool = Field(
            default=False,
            description="Debug mode"
        )
        data_dir: Path = Field(
            default=Path("./data"),
            description="Base data directory"
        )
        temp_dir: Path = Field(
            default=Path("/tmp/auto_repair"),
            description="Temporary files directory"
        )
        
        @root_validator
        def validate_environment(cls, values):
            """Cross-field validation."""
            env = values.get('environment')
            fu = values.get('fu')
            
            # Production safety checks
            if env == 'production':
                if fu and not fu.api_key and not os.getenv('OPENAI_API_KEY'):
                    raise ValueError("API key required in production")
                if values.get('debug'):
                    raise ValueError("Debug must be False in production")
            
            return values
        
        @validator('data_dir', 'temp_dir')
        def create_dirs(cls, v):
            v.mkdir(parents=True, exist_ok=True)
            return v

else:
    # Fallback dataclass implementation if Pydantic unavailable
    @dataclass
    class GBFLConfig:
        formula: str = "ochiai"
        use_gnn: bool = True
        gnn_architecture: str = "graphsage"
        gnn_model_path: Optional[Path] = None
        gnn_hidden_channels: int = 128
        gnn_num_layers: int = 3
        gnn_dropout: float = 0.3
        method_level: bool = True
        max_graph_nodes: int = 10000

    @dataclass
    class REVRConfig:
        enabled: bool = True
        enable_symexec: bool = True
        enable_concolic: bool = True
        smt_solver: str = "z3"
        smt_timeout: int = 30
        max_path_length: int = 100
        max_loop_unroll: int = 5
        abstract_domain: str = "combined"
        widening_threshold: int = 3
        reverse_depth: int = 50
        interval_precision: int = 64
        predicate_set_size: int = 20

    @dataclass
    class PFSConfig:
        enabled: bool = True
        model_path: Optional[Path] = None
        max_patches: int = 100
        max_templates: int = 50
        enable_semantic_synthesis: bool = True
        semantic_timeout: int = 60
        feature_dimensions: int = 50
        template_temperature: float = 0.8
        rank_by: str = "combined"

    @dataclass
    class FUConfig:
        enabled: bool = True
        provider: str = "openai"
        model: str = "gpt-4"
        api_key: Optional[str] = None
        api_base: Optional[str] = None
        temperature: float = 0.3
        max_tokens: int = 2000
        request_timeout: int = 30
        cache_responses: bool = True
        cache_ttl: int = 3600
        max_concurrent_requests: int = 10
        rate_limit_rpm: int = 60

    @dataclass
    class ValidationConfig:
        sandbox_type: str = "docker"
        sandbox_image: str = "auto-repair-sandbox:latest"
        sandbox_timeout: int = 300
        max_memory_mb: int = 2048
        max_cpu_cores: float = 2.0
        network_enabled: bool = False
        required_test_coverage: float = 0.0
        differential_testing: bool = True
        static_analysis_tools: List[str] = field(default_factory=lambda: ["pylint", "mypy"])

    @dataclass
    class OrchestrationConfig:
        max_repair_attempts: int = 3
        human_in_the_loop: bool = True
        auto_apply_confidence_threshold: float = 0.95
        rollback_on_failure: bool = True
        parallel_localization: bool = True
        parallel_synthesis: bool = True
        state_persistence: str = "postgresql"

    @dataclass
    class KnowledgeConfig:
        models_dir: Path = field(default_factory=lambda: Path("./models"))
        embeddings_dir: Path = field(default_factory=lambda: Path("./embeddings"))
        historical_patches_dir: Path = field(default_factory=lambda: Path("./historical_patches"))
        patterns_file: Optional[Path] = None
        enable_online_learning: bool = False
        feedback_retention_days: int = 90

    @dataclass
    class TelemetryConfig:
        log_level: str = "INFO"
        log_format: str = "json"
        metrics_enabled: bool = True
        metrics_backend: str = "prometheus"
        tracing_enabled: bool = True
        jaeger_endpoint: Optional[str] = None
        audit_log_path: Path = field(default_factory=lambda: Path("./audit.log"))

    @dataclass
    class APIConfig:
        host: str = "0.0.0.0"
        port: int = 8080
        workers: int = 4
        cors_origins: List[str] = field(default_factory=lambda: ["*"])
        auth_enabled: bool = True
        jwt_secret: Optional[str] = None
        rate_limit_per_minute: int = 100
        max_request_size_mb: int = 10
        websocket_enabled: bool = True

    @dataclass
    class AutoRepairConfig:
        gbfl: GBFLConfig = field(default_factory=GBFLConfig)
        revr: REVRConfig = field(default_factory=REVRConfig)
        pfs: PFSConfig = field(default_factory=PFSConfig)
        fu: FUConfig = field(default_factory=FUConfig)
        validation: ValidationConfig = field(default_factory=ValidationConfig)
        orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
        knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
        telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
        api: APIConfig = field(default_factory=APIConfig)
        environment: str = "development"
        debug: bool = False
        data_dir: Path = field(default_factory=lambda: Path("./data"))
        temp_dir: Path = field(default_factory=lambda: Path("/tmp/auto_repair"))


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

class ConfigLoader:
    """Load configuration from multiple sources."""
    
    @staticmethod
    def from_file(path: Union[str, Path]) -> AutoRepairConfig:
        """Load from YAML or JSON file."""
        path = Path(path)
        
        with open(path) as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return AutoRepairConfig(**data)
    
    @staticmethod
    def from_env(prefix: str = "AUTO_REPAIR_") -> AutoRepairConfig:
        """Load from environment variables."""
        # Map env vars to config structure
        env_mapping = {
            'ENVIRONMENT': ['environment'],
            'DEBUG': ['debug'],
            'GBFL_FORMULA': ['gbfl', 'formula'],
            'GBFL_USE_GNN': ['gbfl', 'use_gnn'],
            'GBFL_GNN_MODEL_PATH': ['gbfl', 'gnn_model_path'],
            'REVR_ENABLED': ['revr', 'enabled'],
            'REVR_SMT_TIMEOUT': ['revr', 'smt_timeout'],
            'PFS_MODEL_PATH': ['pfs', 'model_path'],
            'PFS_MAX_PATCHES': ['pfs', 'max_patches'],
            'FU_PROVIDER': ['fu', 'provider'],
            'FU_MODEL': ['fu', 'model'],
            'FU_API_KEY': ['fu', 'api_key'],
            'FU_API_BASE': ['fu', 'api_base'],
            'VALIDATION_SANDBOX_TYPE': ['validation', 'sandbox_type'],
            'VALIDATION_SANDBOX_TIMEOUT': ['validation', 'sandbox_timeout'],
            'API_HOST': ['api', 'host'],
            'API_PORT': ['api', 'port'],
            'API_WORKERS': ['api', 'workers'],
        }
        
        data = {}
        for env_var, path in env_mapping.items():
            full_var = f"{prefix}{env_var}"
            if full_var in os.environ:
                value = os.environ[full_var]
                # Type conversion
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                # Set nested value
                current = data
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[path[-1]] = value
        
        return AutoRepairConfig(**data)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> AutoRepairConfig:
        """Load from dictionary."""
        return AutoRepairConfig(**data)
    
    @classmethod
    def auto_load(cls, config_path: Optional[str] = None) -> AutoRepairConfig:
        """
        Auto-load configuration from multiple sources (priority order):
        1. Explicit config file
        2. Environment variables
        3. Default values
        """
        config = AutoRepairConfig()  # Defaults
        
        # Layer 1: Environment variables
        try:
            env_config = cls.from_env()
            config = cls._merge_configs(config, env_config)
        except Exception as e:
            print(f"Warning: Failed to load env config: {e}")
        
        # Layer 2: Config file
        if config_path:
            file_config = cls.from_file(config_path)
            config = cls._merge_configs(config, file_config)
        elif os.getenv('AUTO_REPAIR_CONFIG'):
            file_config = cls.from_file(os.getenv('AUTO_REPAIR_CONFIG'))
            config = cls._merge_configs(config, file_config)
        
        return config
    
    @staticmethod
    def _merge_configs(base: AutoRepairConfig, override: AutoRepairConfig) -> AutoRepairConfig:
        """Merge two config objects, override takes precedence."""
        if PYDANTIC_AVAILABLE:
            base_dict = base.dict()
            override_dict = override.dict(exclude_unset=True)
            merged = {**base_dict, **override_dict}
            return AutoRepairConfig(**merged)
        else:
            # Manual merge for dataclasses
            merged = AutoRepairConfig()
            for field_name in dir(base):
                if not field_name.startswith('_'):
                    base_val = getattr(base, field_name)
                    override_val = getattr(override, field_name, None)
                    if override_val is not None:
                        setattr(merged, field_name, override_val)
                    else:
                        setattr(merged, field_name, base_val)
            return merged


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

# Singleton instance
_config: Optional[AutoRepairConfig] = None

def get_config() -> AutoRepairConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ConfigLoader.auto_load()
    return _config

def set_config(config: AutoRepairConfig):
    """Set global configuration instance."""
    global _config
    _config = config

def reset_config():
    """Reset global configuration."""
    global _config
    _config = None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Default configuration
    config = AutoRepairConfig()
    print("Default config:", config)
    
    # Example 2: From dictionary
    config_dict = {
        "environment": "production",
        "gbfl": {
            "formula": "tarantula",
            "use_gnn": True,
            "gnn_hidden_channels": 256
        },
        "fu": {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "temperature": 0.2
        }
    }
    config = ConfigLoader.from_dict(config_dict)
    print("Dict config:", config)
    
    # Example 3: From environment variables
    # Set some env vars for demo
    os.environ['AUTO_REPAIR_ENVIRONMENT'] = 'staging'
    os.environ['AUTO_REPAIR_GBFL_FORMULA'] = 'jaccard'
    os.environ['AUTO_REPAIR_FU_TEMPERATURE'] = '0.1'
    
    config = ConfigLoader.from_env()
    print("Env config:", config)
    
    # Example 4: YAML config file
    yaml_content = """
    environment: production
    debug: false
    
    gbfl:
      formula: ochiai
      use_gnn: true
      gnn_model_path: /models/gbfl_v2.pt
    
    revr:
      enabled: true
      smt_timeout: 60
      abstract_domain: combined
    
    pfs:
      max_patches: 200
      enable_semantic_synthesis: true
    
    fu:
      provider: openai
      model: gpt-4-turbo
      cache_responses: true
    
    validation:
      sandbox_type: kubernetes
      max_memory_mb: 4096
    
    orchestration:
      human_in_the_loop: true
      auto_apply_confidence_threshold: 0.99
    """
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    config = ConfigLoader.from_file(temp_path)
    print("YAML config:", config)
    
    # Cleanup
    os.unlink(temp_path)
    
    print("\n✅ All configuration examples passed!")

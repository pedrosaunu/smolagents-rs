#[derive(Debug, Clone)]
pub enum AgentError {
    Parsing(String),
    Execution(String),
    MaxSteps(String),
    Generation(String),
}

impl std::error::Error for AgentError {}

impl AgentError {
    pub fn message(&self) -> &str {
        match self {
            Self::Parsing(msg) => msg,
            Self::Execution(msg) => msg,
            Self::MaxSteps(msg) => msg,
            Self::Generation(msg) => msg,
        }
    }
}
impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parsing(msg) => write!(f, "{}", msg),
            Self::Execution(msg) => write!(f, "{}", msg),
            Self::MaxSteps(msg) => write!(f, "{}", msg),
            Self::Generation(msg) => write!(f, "{}", msg),
        }
    }
}

pub type AgentParsingError = AgentError;
pub type AgentExecutionError = AgentError;
pub type AgentMaxStepsError = AgentError;
pub type AgentGenerationError = AgentError;

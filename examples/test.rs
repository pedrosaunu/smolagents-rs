use std::collections::HashMap;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use smolagents::agents::{Agent, FunctionCallingAgent};
use smolagents::models::openai::OpenAIServerModel;
use smolagents::models::ollama::OllamaModelBuilder;
use smolagents::tools::{DuckDuckGoSearchTool, FinalAnswerTool, Tool, ToolGroup, VisitWebsiteTool};

#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The task to execute
    #[arg(short = 't', long)]
    task: String,

    /// The type of agent to use
    #[arg(short = 'a', long, value_enum, default_value = "function-calling")]
    agent_type: AgentType,

    /// List of tools to use
    #[arg(short = 'l', long = "tools", value_enum, num_args = 1.., value_delimiter = ',', default_values_t = [ToolType::DuckDuckGo, ToolType::VisitWebsite])]
    tools: Vec<ToolType>,

    /// OpenAI API key (optional, will use OPENAI_API_KEY env var if not provided)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// OpenAI model ID (optional)
    #[arg(short, long)]
    model: Option<String>,

    /// Whether to stream the output
    #[arg(short, long, default_value = "false")]
    stream: bool,
}

#[derive(Debug)]
enum ToolWrapper {
    FinalAnswer(FinalAnswerTool),
    DuckDuckGo(DuckDuckGoSearchTool),
    VisitWebsite(VisitWebsiteTool),
}

impl Tool for ToolWrapper {
    type Params = serde_json::Value;
    fn name(&self) -> &'static str { 
        match self {
            Self::FinalAnswer(t) => t.name(),
            Self::DuckDuckGo(t) => t.name(),
            Self::VisitWebsite(t) => t.name(),
        }
    }
    fn description(&self) -> &'static str {
        match self {
            Self::FinalAnswer(t) => t.description(),
            Self::DuckDuckGo(t) => t.description(),
            Self::VisitWebsite(t) => t.description(),
        }
    }
    fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> {
        match self {
            Self::FinalAnswer(t) => t.inputs(),
            Self::DuckDuckGo(t) => t.inputs(),
            Self::VisitWebsite(t) => t.inputs(),
        }
    }
    fn output_type(&self) -> &'static str {
        match self {
            Self::FinalAnswer(t) => t.output_type(),
            Self::DuckDuckGo(t) => t.output_type(),
            Self::VisitWebsite(t) => t.output_type(),
        }
    }
    fn is_initialized(&self) -> bool {
        match self {
            Self::FinalAnswer(t) => t.is_initialized(),
            Self::DuckDuckGo(t) => t.is_initialized(),
            Self::VisitWebsite(t) => t.is_initialized(),
        }
    }
    fn forward(&self, args: serde_json::Value) -> Result<String> {
        match self {
            Self::FinalAnswer(t) => Tool::forward(t, serde_json::from_value(args)?),
            Self::DuckDuckGo(t) => Tool::forward(t, serde_json::from_value(args)?),
            Self::VisitWebsite(t) => Tool::forward(t, serde_json::from_value(args)?),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create tools
    let tools = vec![ToolWrapper::FinalAnswer(FinalAnswerTool::new())];

    // Create model
    let model = OpenAIServerModel::new(args.model.as_deref(), None, args.api_key);

    // let model = OllamaModelBuilder::new().model_id("llama3.2").build();

    // Create agent based on type
    let mut agent = match args.agent_type {
        AgentType::FunctionCalling => {
            FunctionCallingAgent::new(model, tools, None, None, Some("CLI Agent"), None)?
        }
    };

    // Run the agent
    let _result = agent.run(&args.task, args.stream, true)?;
    Ok(())
}

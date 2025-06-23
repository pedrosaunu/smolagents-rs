use std::collections::HashMap;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde_json;
use smolagents_rs::agents::Step;
use smolagents_rs::agents::{Agent, CodeAgent, FunctionCallingAgent};
use smolagents_rs::errors::AgentError;
use smolagents_rs::models::model_traits::{Model, ModelResponse};
use smolagents_rs::models::ollama::{OllamaModel, OllamaModelBuilder};
use smolagents_rs::models::openai::OpenAIServerModel;
use smolagents_rs::models::types::Message;
use smolagents_rs::tools::{
    AnyTool, DuckDuckGoSearchTool, GoogleSearchTool, ToolInfo, VisitWebsiteTool,
};
use std::fs::File;

#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
    Code,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
    GoogleSearchTool,
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    OpenAI,
    Ollama,
}

#[derive(Debug)]
enum ModelWrapper {
    OpenAI(OpenAIServerModel),
    Ollama(OllamaModel),
}

enum AgentWrapper {
    FunctionCalling(FunctionCallingAgent<ModelWrapper>),
    Code(CodeAgent<ModelWrapper>),
}

impl AgentWrapper {
    fn run(&mut self, task: &str, stream: bool, reset: bool) -> Result<String> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.run(task, stream, reset),
            AgentWrapper::Code(agent) => agent.run(task, stream, reset),
        }
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.get_logs_mut(),
            AgentWrapper::Code(agent) => agent.get_logs_mut(),
        }
    }
}
impl Model for ModelWrapper {
    fn run(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        match self {
            ModelWrapper::OpenAI(m) => Ok(m.run(messages, tools, max_tokens, args)?),
            ModelWrapper::Ollama(m) => Ok(m.run(messages, tools, max_tokens, args)?),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The type of agent to use
    #[arg(short = 'a', long, value_enum, default_value = "function-calling")]
    agent_type: AgentType,

    /// List of tools to use
    #[arg(short = 'l', long = "tools", value_enum, num_args = 1.., value_delimiter = ',', default_values_t = [ToolType::DuckDuckGo, ToolType::VisitWebsite])]
    tools: Vec<ToolType>,

    /// The type of model to use
    #[arg(short = 'm', long, value_enum, default_value = "open-ai")]
    model_type: ModelType,

    /// OpenAI API key (only required for OpenAI model)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama)
    #[arg(long, default_value = "gpt-4o-mini")]
    model_id: String,

    /// Whether to stream the output
    #[arg(short, long, default_value = "false")]
    stream: bool,

    /// Whether to reset the agent
    #[arg(short, long, default_value = "false")]
    reset: bool,

    /// The task to execute
    #[arg(short, long)]
    task: String,

    /// Base URL for the API
    #[arg(short, long)]
    base_url: Option<String>,
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AnyTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tools: Vec<Box<dyn AnyTool>> = args.tools.iter().map(create_tool).collect();

    // Create model based on type
    let model = match args.model_type {
        ModelType::OpenAI => ModelWrapper::OpenAI(OpenAIServerModel::new(
            args.base_url.as_deref(),
            Some(&args.model_id),
            None,
            args.api_key,
        )),
        ModelType::Ollama => ModelWrapper::Ollama(
            OllamaModelBuilder::new()
                .model_id(&args.model_id)
                .ctx_length(8000)
                .build(),
        ),
    };

    // Create agent based on type
    let mut agent = match args.agent_type {
        AgentType::FunctionCalling => AgentWrapper::FunctionCalling(FunctionCallingAgent::new(
            model,
            tools,
            None,
            None,
            Some("CLI Agent"),
            None,
        )?),
        AgentType::Code => AgentWrapper::Code(CodeAgent::new(
            model,
            tools,
            None,
            None,
            Some("CLI Agent"),
            None,
        )?),
    };

    // Run the agent with the task from stdin
    let _result = agent.run(&args.task, args.stream, args.reset)?;
    let logs = agent.get_logs_mut();

    // store logs in a file
    let mut file = File::create("logs.txt")?;

    // Get the last log entry and serialize it in a controlled way
    for log in logs {
        // Serialize to JSON with pretty printing
        serde_json::to_writer_pretty(&mut file, &log)?;
    }

    // End of example; returning Ok(()) indicates successful execution
    Ok(())
}

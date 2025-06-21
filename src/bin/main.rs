use anyhow::Result;
use clap::{Parser, ValueEnum};
use colored::*;
use smolagents_rs::agents::Step;
use smolagents_rs::agents::{Agent, CodeAgent, FunctionCallingAgent, PlanningAgent};
use smolagents_rs::errors::AgentError;
use smolagents_rs::models::model_traits::{Model, ModelResponse};
use smolagents_rs::models::ollama::{OllamaModel, OllamaModelBuilder};
use smolagents_rs::models::openai::OpenAIServerModel;
use smolagents_rs::models::huggingface::HuggingFaceModel;
use smolagents_rs::models::candle::CandleModel;
use smolagents_rs::models::lightllm::LightLLMModel;
use smolagents_rs::models::types::Message;
use smolagents_rs::tools::{
    AnyTool, DuckDuckGoSearchTool, GoogleSearchTool, RagTool, ToolInfo, VisitWebsiteTool,
    WikipediaSearchTool, TreeSitterTool,
};
use smolagents_rs::sandbox::Sandbox;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};

#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
    Code,
    Planning,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
    GoogleSearchTool,
    WikipediaSearch,
    Rag,
    TreeSitter,
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    OpenAI,
    Ollama,
    HuggingFace,
    Candle,
    LightLLM,
}

#[derive(Debug, Clone)]
enum ModelWrapper {
    OpenAI(OpenAIServerModel),
    Ollama(OllamaModel),
    HuggingFace(HuggingFaceModel),
    Candle(CandleModel),
    LightLLM(LightLLMModel),
}

enum AgentWrapper {
    FunctionCalling(FunctionCallingAgent<ModelWrapper>),
    Code(CodeAgent<ModelWrapper>),
    Planning(PlanningAgent<ModelWrapper>),
}

impl AgentWrapper {
    fn run(&mut self, task: &str, stream: bool, reset: bool) -> Result<String> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.run(task, stream, reset),
            AgentWrapper::Code(agent) => agent.run(task, stream, reset),
            AgentWrapper::Planning(agent) => agent.run(task, stream, reset),
        }
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.get_logs_mut(),
            AgentWrapper::Code(agent) => agent.get_logs_mut(),
            AgentWrapper::Planning(agent) => agent.get_logs_mut(),
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
            ModelWrapper::HuggingFace(m) => Ok(m.run(messages, tools, max_tokens, args)?),
            ModelWrapper::Candle(m) => Ok(m.run(messages, tools, max_tokens, args)?),
            ModelWrapper::LightLLM(m) => Ok(m.run(messages, tools, max_tokens, args)?),
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

    /// API key for the selected model (OpenAI or Hugging Face)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama)
    #[arg(long, default_value = "gpt-4o-mini")]
    model_id: String,

    /// Whether to stream the output
    #[arg(short, long, default_value = "false")]
    stream: bool,

    /// Base URL for the API
    #[arg(short, long)]
    base_url: Option<String>,

    /// Path to the local model directory for Candle
    #[arg(long)]
    model_path: Option<String>,

    /// Run the agent in a sandboxed temporary directory
    #[arg(long, default_value_t = false)]
    sandbox: bool,
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AnyTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
        ToolType::WikipediaSearch => Box::new(WikipediaSearchTool::new()),
        ToolType::Rag => Box::new(RagTool::new(vec![], 3)),
        ToolType::TreeSitter => Box::new(TreeSitterTool::new()),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let _sandbox = if args.sandbox {
        let sb = Sandbox::new()?;
        sb.set_as_cwd()?;
        println!("Using sandbox at {}", sb.path().display());
        Some(sb)
    } else {
        None
    };

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
        ModelType::HuggingFace => ModelWrapper::HuggingFace(HuggingFaceModel::new(
            args.base_url.as_deref(),
            Some(&args.model_id),
            None,
            args.api_key,
        )),
        ModelType::Candle => {
            let path = args
                .model_path
                .clone()
                .unwrap_or_else(|| std::env::var("CANDLE_MODEL_PATH").expect("CANDLE_MODEL_PATH must be set"));
            ModelWrapper::Candle(
                CandleModel::new(&path, None).expect("Failed to load candle model"),
            )
        }
        ModelType::LightLLM => ModelWrapper::LightLLM(LightLLMModel::new(
            args.base_url.as_deref(),
            Some(&args.model_id),
            None,
            args.api_key,
        )),
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
        AgentType::Planning => AgentWrapper::Planning(PlanningAgent::new(
            model,
            tools,
            None,
            None,
            Some("CLI Agent"),
            None,
        )?),
    };

    let mut file: File = File::create("logs.txt")?;

    loop {
        print!("{}", "User: ".yellow().bold());
        io::stdout().flush()?;

        let mut task = String::new();
        io::stdin().read_line(&mut task)?;
        let task = task.trim();

        // Exit if user enters empty line or Ctrl+D
        if task.is_empty() {
            println!("Enter a task to execute");
            continue;
        }
        if task == "exit" {
            break;
        }

        // Run the agent with the task from stdin
        let _result = agent.run(task, args.stream, true)?;
        // Get the last log entry and serialize it in a controlled way

        let logs = agent.get_logs_mut();
        for log in logs {
            // Serialize to JSON with pretty printing
            serde_json::to_writer_pretty(&mut file, &log)?;
        }
    }
    // Successful execution of the CLI
    Ok(())
}

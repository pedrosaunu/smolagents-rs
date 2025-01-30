use anyhow::Result;
use clap::{Parser, ValueEnum};
use smolagents::agents::{Agent, FunctionCallingAgent};
use smolagents::models::openai::OpenAIServerModel;
use smolagents::models::ollama::OllamaModelBuilder;
use smolagents::tools::{DuckDuckGoSearchTool, Tool, VisitWebsiteTool};

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

fn create_tool(tool_type: &ToolType) -> Box<dyn Tool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create tools
    let tools: Vec<Box<dyn Tool>> = args.tools.iter().map(create_tool).collect();

    // Create model
    let model = OpenAIServerModel::new(args.model.as_deref(), None, args.api_key);

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

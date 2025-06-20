use std::sync::Arc;

use smolagents_rs::agents::{Agent, FunctionCallingAgent};
use smolagents_rs::models::openai::OpenAIServerModel;
use smolagents_rs::parallel::run_tasks_parallel;
use smolagents_rs::tools::{AnyTool, DuckDuckGoSearchTool, VisitWebsiteTool};

fn build_agent() -> FunctionCallingAgent<OpenAIServerModel> {
    let tools: Vec<Box<dyn AnyTool>> = vec![
        Box::new(DuckDuckGoSearchTool::new()),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model = OpenAIServerModel::new(
        Some("https://api.openai.com/v1/chat/completions"),
        Some("gpt-4o-mini"),
        None,
        None,
    );
    FunctionCallingAgent::new(model, tools, None, None, None, None).unwrap()
}

fn main() {
    let tasks = vec![
        "What is Rust?".to_string(),
        "Latest news about AI".to_string(),
    ];

    let results = run_tasks_parallel::<FunctionCallingAgent<OpenAIServerModel>>(Arc::new(build_agent), &tasks);

    for (task, result) in tasks.iter().zip(results.into_iter()) {
        println!("Task: {}\nResult: {:?}\n", task, result.unwrap());
    }
}

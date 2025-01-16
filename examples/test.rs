use colored::*;
use log::{self, info, Level, Metadata, Record};
use smolagents::agents::{Agent, FunctionCallingAgent};
use smolagents::models::openai::OpenAIServerModel;
use smolagents::tools::{DuckDuckGoSearchTool, GoogleSearchTool, VisitWebsiteTool};
use std::io::Write;

struct ColoredLogger;

impl log::Log for ColoredLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let mut stdout = std::io::stdout();
            let msg = record.args().to_string();

            // Add a newline before each message for spacing
            writeln!(stdout).unwrap();

            // Check for specific prefixes and apply different colors
            if msg.starts_with("Observation:") {
                let (prefix, content) = msg.split_at(12);
                writeln!(stdout, "{}{}", prefix.yellow().bold(), content.green()).unwrap();
            } else if msg.starts_with("Executing tool call:") {
                let (prefix, content) = msg.split_at(21);
                writeln!(stdout, "{}{}", prefix.magenta().bold(), content.cyan()).unwrap();
            } else if msg.starts_with("Plan:") {
                let (prefix, content) = msg.split_at(5);
                writeln!(stdout, "{}{}", prefix.red().bold(), content.blue().italic()).unwrap();
            } else if msg.starts_with("Final answer:") {
                let (prefix, content) = msg.split_at(13);
                writeln!(
                    stdout,
                    "\n{}{}\n",
                    prefix.green().bold(),
                    content.white().bold()
                )
                .unwrap();
            } else {
                writeln!(stdout, "{}", msg.blue()).unwrap();
            }
        }
    }

    fn flush(&self) {}
}

static LOGGER: ColoredLogger = ColoredLogger;

fn main() {
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Info);

    info!("Starting agent");

    let mut agent = FunctionCallingAgent::new(
        OpenAIServerModel::new(None, None, None),
        vec![
            Box::new(DuckDuckGoSearchTool::new()),
            Box::new(VisitWebsiteTool::new()),
        ],
        None,
        None,
        Some("multistep_agent"),
        None,
    )
    .unwrap();

    agent
        .run("Whats the weather in Eindhoven?", false, true)
        .unwrap();
}

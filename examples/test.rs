
use log::{self, info, Record, Level, Metadata};
use smolagents::agents:: MultiStepAgent;
use smolagents::models::OpenAIServerModel;
use smolagents::tools::VisitWebsiteTool;
use colored::*;
use std::io::Write;

struct ColoredLogger;

impl log::Log for ColoredLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let mut stdout = std::io::stdout();
            if record.level() == Level::Info {
                writeln!(stdout, "{}", record.args().to_string().blue()).unwrap();
            } else {
                writeln!(stdout, "{}", record.args()).unwrap();
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

    let mut agent = MultiStepAgent::new(
        OpenAIServerModel::new(None, None, None),
        vec![VisitWebsiteTool::new()],
        None,
        None,
        Some("multistep_agent"),
        None,
    )
    .unwrap();

    println!("{}", agent.run("Who is akshay ballal? visit https://www.akshaymakes.com/", false, true).unwrap());
}

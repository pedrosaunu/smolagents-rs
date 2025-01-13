use std::collections::HashMap;

use smolagents::agents::{get_tool_description_with_args, Agent, MultiStepAgent};
use smolagents::models::OpenAIServerModel;
use smolagents::tools::VisitWebsiteTool;
fn main() {
    
    let agent = MultiStepAgent::new(
        OpenAIServerModel::new(None, None, None),
        vec![VisitWebsiteTool::new()],
        None,
        None,
        Some("multistep_agent"),
        None,
    )
    .unwrap();

    // let agent2 = MultiStepAgent::new(
    //     vec![VisitWebsiteTool::new()],
    //     None,
    //     Some(HashMap::from([(
    //         "agent".to_string(),
    //         Box::new(agentic) as Box<dyn Agent>,
    //     )])),
    //     None,
    // )
    // .unwrap();
    // println!("{}", agent2.system_prompt_template);
    println!(
        "{:?}",
        agent
            .execute_tool_call(
                "visit_website",
                HashMap::from([("url".to_string(), "https://www.akshaymakes.com/".to_string())])
            )
            .unwrap()
    );
}

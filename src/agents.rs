use crate::models::{MessageRole, Model};
use crate::prompts::{user_prompt_plan, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN};
use crate::tools::Tool;
use crate::{models::Message, prompts::CODE_SYSTEM_PROMPT};
use std::collections::HashMap;

use anyhow::{Error as E, Result};
use log::{error, info, warn};
const DEFAULT_TOOL_DESCRIPTION_TEMPLATE: &str = r#"
{{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
"#;

use std::fmt::Debug;

pub trait Agent: Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> String {
        "".to_string()
    }
}

#[derive(Debug)]
enum Step {
    PlanningStep(String, String),
    TaskStep(String),
    SystemPromptStep(String),
    ActionStep(AgentStep),
    ToolCall(ToolCall),
}

#[derive(Debug)]
pub struct AgentStep {
    agent_memory: Option<Vec<Step>>,
    llm_output: Option<String>,
    tool_call: Option<ToolCall>,
    error: Option<AgentError>,
    observations: Option<String>,
    step: usize,
}

#[derive(Debug, Clone)]
pub struct AgentError {
    pub message: String,
}

impl From<E> for AgentError {
    fn from(error: E) -> Self {
        error!("{}", error);
        AgentError {
            message: error.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    name: String,
    arguments: HashMap<String, String>,
    id: String,
}

#[derive(Debug)]
pub struct MultiStepAgent<T: Tool, M: Model<T>> {
    pub model: M,
    pub tools: HashMap<String, T>,
    pub system_prompt_template: String,
    pub name: &'static str,
    pub managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
    pub description: String,
    pub max_steps: usize,
    pub step_number: usize,
    pub task: String,
    logs: Vec<Step>,
}

impl<T: Tool + Debug, M: Model<T>> Agent for MultiStepAgent<T, M> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn description(&self) -> String {
        self.description.clone()
    }
}

impl<T: Tool, M: Model<T>> MultiStepAgent<T, M> {
    pub fn new(
        model: M,
        tools: Vec<T>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let name = "MultiStepAgent";
        let system_prompt_template = match system_prompt {
            Some(prompt) => prompt.to_string(),
            None => {
                let tool_refs: Vec<&T> = tools.iter().collect();
                format_prompt_with_tools(&tool_refs, CODE_SYSTEM_PROMPT)
            }
        };
        let description = match description {
            Some(desc) => desc.to_string(),
            None => "A multi-step agent that can solve tasks using a series of tools".to_string(),
        };

        let tools = tools
            .into_iter()
            .map(|tool| (tool.name().to_string(), tool))
            .collect();

        let mut agent = MultiStepAgent {
            model,
            tools,
            system_prompt_template,
            name,
            managed_agents,
            description,
            max_steps: max_steps.unwrap_or(10),
            step_number: 0,
            task: "".to_string(),
            logs: Vec::new(),
        };

        agent.initialize_system_prompt()?;
        Ok(agent)
    }

    fn initialize_system_prompt(&mut self) -> Result<String> {
        let tools: Vec<&T> = self.tools.values().collect();
        let tools: Vec<&T> = tools.iter().map(|&tool| tool).collect();
        self.system_prompt_template =
            format_prompt_with_tools(&tools, &self.system_prompt_template);
        match &self.managed_agents {
            Some(managed_agents) => {
                self.system_prompt_template = format_prompt_with_managed_agent_description(
                    self.system_prompt_template.clone(),
                    managed_agents,
                    None,
                )?;
            }
            None => {
                self.system_prompt_template = format_prompt_with_managed_agent_description(
                    self.system_prompt_template.clone(),
                    &HashMap::new(),
                    None,
                )?;
            }
        }
        Ok(self.system_prompt_template.clone())
    }

    pub fn write_inner_memory_from_logs(self, summary_mode: Option<bool>) -> Vec<Message> {
        let mut memory = Vec::new();
        let summary_mode = summary_mode.unwrap_or(false);
        for log in self.logs {
            match log {
                Step::ToolCall(_) => {}
                Step::PlanningStep(plan, facts) => {
                    memory.push(Message {
                        role: MessageRole::Assistant,
                        content: "[PLAN]:\n".to_owned() + plan.as_str(),
                    });

                    if !summary_mode {
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: "[FACTS]:\n".to_owned() + facts.as_str(),
                        });
                    }
                }
                Step::TaskStep(task) => {
                    memory.push(Message {
                        role: MessageRole::User,
                        content: "New Task".to_owned() + task.as_str(),
                    });
                }
                Step::SystemPromptStep(prompt) => {
                    memory.push(Message {
                        role: MessageRole::System,
                        content: prompt,
                    });
                }
                Step::ActionStep(step_log) => {
                    if step_log.llm_output.is_some() && !summary_mode {
                        memory.push(Message {
                            role: MessageRole::System,
                            content: step_log.llm_output.unwrap(),
                        });
                    }
                    if step_log.tool_call.is_some() {
                        let tool_call_message = Message {
                            role: MessageRole::Assistant,
                            content: format!(
                                r#"[
                                \{{
                                \'id\': \"{}\"
                                \'type\': \"function\",
                                \'function\": {{
                                    \"name\": \"{}\"
                                    \"arguments\": {:?}
                            }}
                                ]"#,
                                step_log.tool_call.clone().unwrap().id,
                                step_log.tool_call.clone().unwrap().name,
                                step_log.tool_call.clone().unwrap().arguments
                            ),
                        };
                        memory.push(tool_call_message);
                    }
                    if step_log.tool_call.is_none() && step_log.error.is_some() {
                        let message_content = "Error: ".to_owned() + step_log.error.clone().unwrap().message.as_str()+"\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: message_content,
                        });
                    }
                    if step_log.tool_call.is_some()
                        && (step_log.error.is_some() || step_log.observations.is_some())
                    {
                        let mut message_content = "".to_string();
                        if step_log.error.is_some() {
                            message_content = "Error: ".to_owned() + step_log.error.unwrap().message.as_str()+"\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        } else if step_log.observations.is_some() {
                            message_content = "Observations:\n".to_owned()
                                + step_log.observations.unwrap().as_str();
                        }
                        let tool_response_message = {
                            Message {
                                role: MessageRole::ToolResponse,
                                content: format!(
                                    "Call id: {}\n{}",
                                    step_log.tool_call.unwrap().id,
                                    message_content
                                ),
                            }
                        };
                        memory.push(tool_response_message);
                    }
                }
            }
        }
        memory
    }
    pub fn execute_tool_call(
        &self,
        tool_name: &str,
        arguments: HashMap<String, String>,
    ) -> Result<String> {
        let tool = self.tools.get(tool_name).unwrap();
        let output = tool.forward(arguments)?;
        let output_str = output.downcast_ref::<String>().unwrap();
        Ok(output_str.clone())
    }

    pub fn run(&mut self, task: &str, stream: bool) -> Result<String> {
        self.task = task.to_string();
        match stream {
            true => self.stream_run(self.task.as_str()),
            false => self.direct_run(self.task.as_str()),
        }
    }

    fn stream_run(&self, task: &str) -> Result<String> {
        todo!()
    }

    fn step(&self, step_log:Step)->Result<String> {
        todo!()
    }

    fn direct_run(&self, task: &str) -> Result<String> {
        let mut final_answer: Option<String> = None;
        while final_answer == None && self.step_number < self.max_steps {
            let step_log = Step::ActionStep(AgentStep {
                agent_memory: None,
                llm_output: None,
                tool_call: None,
                error: None,
                observations: None,
                step: self.step_number,
            });
            final_answer = Some(self.step(step_log)?);
        }
        todo!()
    }

    pub fn planning_step(&mut self, task: &str, is_first_step: bool, step: usize) -> () {
        match is_first_step {
            true => {
                let message_prompt_facts = Message {
                    role: MessageRole::System,
                    content: SYSTEM_PROMPT_FACTS.to_string(),
                };
                let message_prompt_task = Message {
                    role: MessageRole::User,
                    content: format!(
                        "Here is the task: ```
                    {}
                    ```
                    Now Begin!
                    ",
                        task
                    ),
                };

                let answer_facts = self
                    .model
                    .run(
                        vec![message_prompt_facts, message_prompt_task],
                        vec![],
                        None,
                        None,
                    )
                    .unwrap_or("".to_string());
                let message_system_prompt_plan = Message {
                    role: MessageRole::System,
                    content: SYSTEM_PROMPT_PLAN.to_string(),
                };
                let tool_descriptions =
                    get_tool_descriptions(&self.tools.values().collect::<Vec<&T>>()).join("\n");
                let message_user_prompt_plan = Message {
                    role: MessageRole::User,
                    content: user_prompt_plan(
                        task,
                        &tool_descriptions,
                        &show_agents_description(
                            self.managed_agents.as_ref().unwrap_or(&HashMap::new()),
                        ),
                        &answer_facts,
                    ),
                };
                let answer_plan = self
                    .model
                    .run(
                        vec![message_system_prompt_plan, message_user_prompt_plan],
                        vec![],
                        None,
                        Some(HashMap::from([(
                            "stop_sequences".to_string(),
                            "<end_plan>".to_string(),
                        )])),
                    )
                    .unwrap();
                let final_plan_redaction = format!(
                    "Here is the plan of action that I will follow for the task: \n{}",
                    answer_plan
                );
                let final_facts_redaction =
                    format!("Here are the facts that I know so far: \n{}", answer_facts);
                self.logs.push(Step::PlanningStep(
                    final_plan_redaction.clone(),
                    final_facts_redaction,
                ));
                info!("Plan: {}", final_plan_redaction);
                ()
            }
            false => (),
        };
    }
}

pub fn get_tool_description_with_args(tool: &dyn Tool) -> String {
    let mut description = DEFAULT_TOOL_DESCRIPTION_TEMPLATE.to_string();
    description = description.replace("{{ tool.name }}", tool.name());
    description = description.replace("{{ tool.description }}", tool.description());

    let inputs_description: Vec<String> = tool
        .inputs()
        .iter()
        .map(|(key, value)| {
            let type_desc = value.get("type").unwrap();
            let desc = value.get("description").unwrap();
            // .downcast_ref::<&str>()
            // .unwrap();
            format!("{} ({}): {}", key, type_desc, desc)
        })
        .collect();

    description = description.replace("{{tool.inputs}}", &inputs_description.join(", "));
    description = description.replace("{{tool.output_type}}", tool.output_type());

    description
}

pub fn get_tool_descriptions<T: Tool>(tools: &[&T]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| get_tool_description_with_args(*tool))
        .collect()
}
pub fn format_prompt_with_tools<'a, T: Tool>(tools: &'a [&T], prompt_template: &'a str) -> String {
    let tool_descriptions = get_tool_descriptions(tools);
    let mut prompt = prompt_template.to_string();
    prompt = prompt.replace("{{tool_descriptions}}", &tool_descriptions.join("\n"));
    if prompt.contains("{{tool_names}}") {
        let tool_names: Vec<String> = tools.iter().map(|tool| tool.name().to_string()).collect();
        prompt = prompt.replace("{{tool_names}}", &tool_names.join(", "));
    }
    prompt
}

pub fn show_agents_description(managed_agents: &HashMap<String, Box<dyn Agent>>) -> String {
    let mut managed_agent_description = r#"You can also give requests to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'request', a long string explaining your request.
Given that this team member is a real human, you should be very verbose in your request.
Here is a list of the team members that you can call:"#.to_string();

    for (name, agent) in managed_agents.iter() {
        managed_agent_description.push_str(&format!("{}: {:?}\n", name, agent.description()));
    }

    managed_agent_description
}

pub fn format_prompt_with_managed_agent_description(
    prompt_template: String,
    managed_agents: &HashMap<String, Box<dyn Agent>>,
    agent_descriptions_placeholder: Option<&str>,
) -> Result<String> {
    let agent_descriptions_placeholder = match agent_descriptions_placeholder {
        Some(placeholder) => placeholder,
        None => "{{managed_agents_descriptions}}",
    };
    if !prompt_template.contains(agent_descriptions_placeholder) {
        return Err(E::msg("The prompt template does not contain the placeholder for the managed agents descriptions"));
    }

    if managed_agents.keys().len() > 0 {
        return Ok(prompt_template.replace(
            agent_descriptions_placeholder,
            &show_agents_description(managed_agents),
        ));
    } else {
        return Ok(prompt_template.replace(agent_descriptions_placeholder, ""));
    }
}

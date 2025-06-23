//! This module contains the agents that can be used to solve tasks.
//!
//! Currently, there are three agents:
//! - The function calling agent. This agent is used for models that have tool calling capabilities.
//! - The code agent. This agent takes tools and can write simple python code that is executed to solve the task.
//! - The planning agent. This agent first creates a high level plan and then executes it using the function calling agent.
//!
//! To use this agent you need to enable the `code-agent` feature.
//!
//! You can also implement your own agents by implementing the `Agent` trait.
//!
//!
use crate::errors::AgentError;
use crate::models::model_traits::Model;
use crate::models::openai::ToolCall;
use crate::models::types::Message;
use crate::models::types::MessageRole;
use crate::prompts::{
    user_prompt_plan, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN, TOOL_CALLING_SYSTEM_PROMPT,
};
use crate::tools::{AnyTool, FinalAnswerTool, ToolGroup, ToolInfo};
use std::collections::HashMap;

use anyhow::Result;
use colored::Colorize;
use log::info;

use serde::Serialize;
use serde_json::json;
#[cfg(feature = "code-agent")]
use {
    crate::errors::InterpreterError, crate::local_python_interpreter::LocalPythonInterpreter,
    crate::models::openai::FunctionCall, crate::prompts::CODE_SYSTEM_PROMPT, regex::Regex,
};

const DEFAULT_TOOL_DESCRIPTION_TEMPLATE: &str = r#"
{{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
"#;

use std::fmt::Debug;

pub fn get_tool_description_with_args(tool: &ToolInfo) -> String {
    let mut description = DEFAULT_TOOL_DESCRIPTION_TEMPLATE.to_string();
    description = description.replace("{{ tool.name }}", tool.function.name);
    description = description.replace("{{ tool.description }}", tool.function.description);
    description = description.replace(
        "{{tool.inputs}}",
        json!(&tool.function.parameters.schema)["properties"]
            .to_string()
            .as_str(),
    );

    description
}

pub fn get_tool_descriptions(tools: &[ToolInfo]) -> Vec<String> {
    tools.iter().map(get_tool_description_with_args).collect()
}
pub fn format_prompt_with_tools(tools: Vec<ToolInfo>, prompt_template: &str) -> String {
    let tool_descriptions = get_tool_descriptions(&tools);
    let mut prompt = prompt_template.to_string();
    prompt = prompt.replace("{{tool_descriptions}}", &tool_descriptions.join("\n"));
    if prompt.contains("{{tool_names}}") {
        let tool_names: Vec<String> = tools
            .iter()
            .map(|tool| tool.function.name.to_string())
            .collect();
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
    let agent_descriptions_placeholder =
        agent_descriptions_placeholder.unwrap_or("{{managed_agents_descriptions}}");

    if managed_agents.keys().len() > 0 {
        Ok(prompt_template.replace(
            agent_descriptions_placeholder,
            &show_agents_description(managed_agents),
        ))
    } else {
        Ok(prompt_template.replace(agent_descriptions_placeholder, ""))
    }
}

pub trait Agent {
    fn name(&self) -> &'static str;
    fn get_max_steps(&self) -> usize;
    fn get_step_number(&self) -> usize;
    fn reset_step_number(&mut self);
    fn increment_step_number(&mut self);
    fn get_logs_mut(&mut self) -> &mut Vec<Step>;
    fn set_task(&mut self, task: &str);
    fn get_system_prompt(&self) -> &str;
    fn description(&self) -> String {
        "".to_string()
    }
    fn model(&self) -> &dyn Model;
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>>;
    fn direct_run(&mut self, _task: &str) -> Result<String> {
        let mut final_answer: Option<String> = None;
        while final_answer.is_none() && self.get_step_number() < self.get_max_steps() {
            println!("Step number: {:?}", self.get_step_number());
            let mut step_log = Step::ActionStep(AgentStep {
                agent_memory: None,
                llm_output: None,
                tool_call: None,
                error: None,
                observations: None,
                _step: self.get_step_number(),
            });

            final_answer = self.step(&mut step_log)?;
            self.get_logs_mut().push(step_log);
            self.increment_step_number();
        }

        if final_answer.is_none() && self.get_step_number() >= self.get_max_steps() {
            final_answer = self.provide_final_answer(_task)?;
        }
        info!(
            "Final answer: {}",
            final_answer
                .clone()
                .unwrap_or("Could not find answer".to_string())
        );
        Ok(final_answer.unwrap_or_else(|| "Max steps reached without final answer".to_string()))
    }
    fn stream_run(&mut self, _task: &str) -> Result<String> {
        self.direct_run(_task)
    }
    fn run(&mut self, task: &str, stream: bool, reset: bool) -> Result<String> {
        // self.task = task.to_string();
        self.set_task(task);

        let system_prompt_step = Step::SystemPromptStep(self.get_system_prompt().to_string());
        if reset {
            self.get_logs_mut().clear();
            self.get_logs_mut().push(system_prompt_step);
            self.reset_step_number();
        } else if self.get_logs_mut().is_empty() {
            self.get_logs_mut().push(system_prompt_step);
        } else {
            self.get_logs_mut()[0] = system_prompt_step;
        }
        self.get_logs_mut().push(Step::TaskStep(task.to_string()));
        match stream {
            true => self.stream_run(task),
            false => self.direct_run(task),
        }
    }
    fn provide_final_answer(&mut self, task: &str) -> Result<Option<String>> {
        let mut input_messages = vec![Message {
            role: MessageRole::System,
            content: "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:".to_string(),
        }];

        input_messages.extend(self.write_inner_memory_from_logs(Some(true))?[1..].to_vec());
        input_messages.push(Message {
            role: MessageRole::User,
            content: format!("Based on the above, please provide an answer to the following user request: \n```\n{}", task),
        });
        let response = self
            .model()
            .run(input_messages, vec![], None, None)?
            .get_response()?;
        Ok(Some(response))
    }

    fn write_inner_memory_from_logs(&mut self, summary_mode: Option<bool>) -> Result<Vec<Message>> {
        let mut memory = Vec::new();
        let summary_mode = summary_mode.unwrap_or(false);
        for log in self.get_logs_mut() {
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
                        content: "New Task: ".to_owned() + task.as_str(),
                    });
                }
                Step::SystemPromptStep(prompt) => {
                    memory.push(Message {
                        role: MessageRole::System,
                        content: prompt.to_string(),
                    });
                }
                Step::ActionStep(step_log) => {
                    if step_log.llm_output.is_some() && !summary_mode {
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: step_log.llm_output.clone().unwrap_or_default(),
                        });
                    }
                    if step_log.tool_call.is_some() {
                        let tool_call_message = step_log
                            .tool_call
                            .clone()
                            .unwrap()
                            .iter()
                            .map(|tool_call| -> Message {
                                Message {
                                    role: MessageRole::Assistant,
                                    content: serde_json::to_string_pretty(&tool_call)
                                        .unwrap_or_default(),
                                }
                            })
                            .collect::<Vec<_>>();
                        memory.extend(tool_call_message);
                    }

                    if let (Some(tool_calls), Some(observations)) =
                        (&step_log.tool_call, &step_log.observations)
                    {
                        for (i, tool_call) in tool_calls.iter().enumerate() {
                            let message_content = format!(
                                "Call id: {}\nObservation: {}",
                                tool_call.id.as_deref().unwrap_or_default(),
                                observations[i]
                            );

                            memory.push(Message {
                                role: MessageRole::User,
                                content: message_content,
                            });
                        }
                    } else if let Some(observations) = &step_log.observations {
                        memory.push(Message {
                            role: MessageRole::User,
                            content: format!("Observations: {}", observations.join("\n")),
                        });
                    }
                    if step_log.error.is_some() {
                        let error_string =
                            "Error: ".to_owned() + step_log.error.clone().unwrap().message(); // Its fine to unwrap because we check for None above

                        let error_string = error_string + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        memory.push(Message {
                            role: MessageRole::User,
                            content: error_string,
                        });
                    }
                }
            }
        }
        Ok(memory)
    }
}

#[derive(Debug, Serialize, Clone)]
pub enum Step {
    PlanningStep(String, String),
    TaskStep(String),
    SystemPromptStep(String),
    ActionStep(AgentStep),
    ToolCall(ToolCall),
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::PlanningStep(plan, facts) => {
                write!(f, "PlanningStep(plan: {}, facts: {})", plan, facts)
            }
            Step::TaskStep(task) => write!(f, "TaskStep({})", task),
            Step::SystemPromptStep(prompt) => write!(f, "SystemPromptStep({})", prompt),
            Step::ActionStep(step) => write!(f, "ActionStep({})", step),
            Step::ToolCall(tool_call) => write!(f, "ToolCall({:?})", tool_call),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentStep {
    agent_memory: Option<Vec<Message>>,
    llm_output: Option<String>,
    tool_call: Option<Vec<ToolCall>>,
    error: Option<AgentError>,
    observations: Option<Vec<String>>,
    _step: usize,
}

impl std::fmt::Display for AgentStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentStep({:?})", self)
    }
}

// Define a trait for the parent functionality

pub struct MultiStepAgent<M: Model> {
    pub model: M,
    pub tools: Vec<Box<dyn AnyTool>>,
    pub system_prompt_template: String,
    pub name: &'static str,
    pub managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
    pub description: String,
    pub max_steps: usize,
    pub step_number: usize,
    pub task: String,
    pub input_messages: Option<Vec<Message>>,
    pub logs: Vec<Step>,
}

impl<M: Model + Debug> Agent for MultiStepAgent<M> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn get_max_steps(&self) -> usize {
        self.max_steps
    }
    fn get_step_number(&self) -> usize {
        self.step_number
    }
    fn set_task(&mut self, task: &str) {
        self.task = task.to_string();
    }
    fn get_system_prompt(&self) -> &str {
        &self.system_prompt_template
    }
    fn increment_step_number(&mut self) {
        self.step_number += 1;
    }
    fn reset_step_number(&mut self) {
        self.step_number = 0;
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        &mut self.logs
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn model(&self) -> &dyn Model {
        &self.model
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.write_inner_memory_from_logs(None)?;
                self.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());

                let tools = self
                    .tools
                    .iter()
                    .map(|tool| tool.tool_info())
                    .collect::<Vec<_>>();

                let model_message = self.model.run(
                    self.input_messages.as_ref().unwrap().clone(),
                    tools,
                    None,
                    Some(HashMap::from([(
                        "stop".to_string(),
                        vec!["Observation:".to_string()],
                    )])),
                )?;

                let mut observations = Vec::new();
                let tools = model_message.get_tools_used()?;
                step_log.tool_call = Some(tools.clone());

                if let Ok(response) = model_message.get_response() {
                    if !response.trim().is_empty() {
                        observations.push(response.clone());
                    }
                    if tools.is_empty() {
                        return Ok(Some(response));
                    }
                }

                for tool in tools {
                    let function_name = tool.clone().function.name;
                    match function_name.as_str() {
                        "final_answer" => {
                            info!("Executing tool call: {}", function_name);
                            let answer = self.tools.call(&tool.function)?;
                            self.write_inner_memory_from_logs(None)?;
                            return Ok(Some(answer));
                        }
                        _ => {
                            info!(
                                "Executing tool call: {} with arguments: {:?}",
                                function_name, tool.function.arguments
                            );
                            let observation_res = self.tools.call(&tool.function);
                            match observation_res {
                                Ok(mut observation) => {
                                    if let Some(answer) = detect_final_answer(&observation) {
                                        return Ok(Some(answer));
                                    }
                                    if observation.len() > 30000 {
                                        observation = truncate_observation(&observation, 30000);
                                    }
                                    observations.push(format!(
                                        "Observation from {}: {}",
                                        function_name, observation
                                    ));
                                }
                                Err(e) => {
                                    observations.push(e.to_string());
                                    info!("Error: {}", e);
                                }
                            }
                        }
                    }
                }

                step_log.observations = Some(observations);
                let summary = truncate_observation(
                    &step_log
                        .observations
                        .clone()
                        .unwrap_or_default()
                        .join("\n")
                        .trim(),
                    30000,
                );
                info!("Observation: {}", summary);
                Ok(None)
            }
            Step::ToolCall(tool_call) => {
                let function_name = tool_call.function.name.clone();
                info!("Executing tool call: {}", function_name);
                let observation = self.tools.call(&tool_call.function)?;
                info!("Observation: {}", truncate_observation(&observation, 30000));
                Ok(Some(observation))
            }
            Step::PlanningStep(..) | Step::TaskStep(..) | Step::SystemPromptStep(..) => Ok(None),
        }
    }
}

impl<M: Model> MultiStepAgent<M> {
    pub fn new(
        model: M,
        mut tools: Vec<Box<dyn AnyTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        // Initialize logger
        crate::logger::init_logger_from_env();

        let name = "MultiStepAgent";

        let system_prompt_template = match system_prompt {
            Some(prompt) => prompt.to_string(),
            None => TOOL_CALLING_SYSTEM_PROMPT.to_string(),
        };
        let description = match description {
            Some(desc) => desc.to_string(),
            None => "A multi-step agent that can solve tasks using a series of tools".to_string(),
        };

        let final_answer_tool = FinalAnswerTool::new();
        tools.push(Box::new(final_answer_tool));

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
            input_messages: None,
        };

        agent.initialize_system_prompt()?;
        Ok(agent)
    }

    fn initialize_system_prompt(&mut self) -> Result<String> {
        let tools = self.tools.tool_info();
        self.system_prompt_template = format_prompt_with_tools(tools, &self.system_prompt_template);
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
        self.system_prompt_template = self
            .system_prompt_template
            .replace("{{current_time}}", &chrono::Local::now().to_string());
        Ok(self.system_prompt_template.clone())
    }

    pub fn planning_step(&mut self, task: &str, is_first_step: bool, _step: usize) {
        if is_first_step {
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
                .unwrap()
                .get_response()
                .unwrap_or("".to_string());
            let message_system_prompt_plan = Message {
                role: MessageRole::System,
                content: SYSTEM_PROMPT_PLAN.to_string(),
            };
            let tool_descriptions = serde_json::to_string(
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.tool_info())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
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
                        "stop".to_string(),
                        vec!["Observation:".to_string()],
                    )])),
                )
                .unwrap()
                .get_response()
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
            info!("Plan: {}", final_plan_redaction.blue().bold());
        }
    }
}

pub struct FunctionCallingAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
}

impl<M: Model + Debug> FunctionCallingAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AnyTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(TOOL_CALLING_SYSTEM_PROMPT);
        let base_agent = MultiStepAgent::new(
            model,
            tools,
            Some(system_prompt),
            managed_agents,
            description,
            max_steps,
        )?;
        Ok(Self { base_agent })
    }

    fn step_stream(
        &mut self,
        log_entry: &mut Step,
        callback: &mut dyn FnMut(&str),
    ) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                let tools = self
                    .base_agent
                    .tools
                    .iter()
                    .map(|tool| tool.tool_info())
                    .collect::<Vec<_>>();
                let model_message = self.base_agent.model.run_stream(
                    self.base_agent.input_messages.as_ref().unwrap().clone(),
                    tools,
                    None,
                    Some(HashMap::from([(
                        "stop".to_string(),
                        vec!["Observation:".to_string()],
                    )])),
                    callback,
                )?;

                let mut observations = Vec::new();
                let tools = model_message.get_tools_used()?;
                step_log.tool_call = Some(tools.clone());

                if let Ok(response) = model_message.get_response() {
                    if !response.trim().is_empty() {
                        observations.push(response.clone());
                    }
                    if tools.is_empty() {
                        return Ok(Some(response));
                    }
                }
                for tool in tools {
                    let function_name = tool.clone().function.name;

                    match function_name.as_str() {
                        "final_answer" => {
                            info!("Executing tool call: {}", function_name);
                            let answer = self.base_agent.tools.call(&tool.function)?;
                            self.base_agent.write_inner_memory_from_logs(None)?;
                            return Ok(Some(answer));
                        }
                        _ => {
                            info!(
                                "Executing tool call: {} with arguments: {:?}",
                                function_name, tool.function.arguments
                            );
                            let observation_res = self.base_agent.tools.call(&tool.function);
                            match observation_res {
                                Ok(mut observation) => {
                                    if let Some(answer) = detect_final_answer(&observation) {
                                        return Ok(Some(answer));
                                    }
                                    if observation.len() > 30000 {
                                        observation = truncate_observation(&observation, 30000);
                                    }
                                    observations.push(format!(
                                        "Observation from {}: {}",
                                        function_name, observation
                                    ));
                                }
                                Err(e) => {
                                    observations.push(e.to_string());
                                    info!("Error: {}", e);
                                }
                            }
                        }
                    }
                }
                step_log.observations = Some(observations);

                let summary = truncate_observation(
                    &step_log
                        .observations
                        .clone()
                        .unwrap_or_default()
                        .join("\n")
                        .trim(),
                    30000,
                );
                info!(
                    "Observation: {} \n ....This content has been truncated due to the 30000 character limit.....",
                    summary
                );
                Ok(None)
            }
            Step::ToolCall(tool_call) => {
                let function_name = tool_call.function.name.clone();
                info!("Executing tool call: {}", function_name);
                let observation = self.base_agent.tools.call(&tool_call.function)?;
                info!("Observation: {}", truncate_observation(&observation, 30000));
                Ok(Some(observation))
            }
            Step::PlanningStep(plan, facts) => {
                info!("Plan:\n{}", truncate_observation(plan, 30000));
                info!("Facts:\n{}", truncate_observation(facts, 30000));
                Ok(None)
            }
            Step::TaskStep(task) => {
                info!("Task: {}", task);
                Ok(None)
            }
            Step::SystemPromptStep(prompt) => {
                info!("System prompt: {}", truncate_observation(prompt, 30000));
                Ok(None)
            }
        }
    }
}

impl<M: Model + Debug> Agent for FunctionCallingAgent<M> {
    fn name(&self) -> &'static str {
        self.base_agent.name()
    }
    fn set_task(&mut self, task: &str) {
        self.base_agent.set_task(task);
    }
    fn get_system_prompt(&self) -> &str {
        self.base_agent.get_system_prompt()
    }
    fn get_max_steps(&self) -> usize {
        self.base_agent.get_max_steps()
    }
    fn get_step_number(&self) -> usize {
        self.base_agent.get_step_number()
    }
    fn reset_step_number(&mut self) {
        self.base_agent.reset_step_number();
    }
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number();
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                let tools = self
                    .base_agent
                    .tools
                    .iter()
                    .map(|tool| tool.tool_info())
                    .collect::<Vec<_>>();
                let model_message = self.base_agent.model.run(
                    self.base_agent.input_messages.as_ref().unwrap().clone(),
                    tools,
                    None,
                    Some(HashMap::from([(
                        "stop".to_string(),
                        vec!["Observation:".to_string()],
                    )])),
                )?;

                let mut observations = Vec::new();
                let tools = model_message.get_tools_used()?;
                step_log.tool_call = Some(tools.clone());

                if let Ok(response) = model_message.get_response() {
                    if !response.trim().is_empty() {
                        observations.push(response.clone());
                    }
                    if tools.is_empty() {
                        return Ok(Some(response));
                    }
                }
                for tool in tools {
                    let function_name = tool.clone().function.name;

                    match function_name.as_str() {
                        "final_answer" => {
                            info!("Executing tool call: {}", function_name);
                            let answer = self.base_agent.tools.call(&tool.function)?;
                            self.base_agent.write_inner_memory_from_logs(None)?;
                            return Ok(Some(answer));
                        }
                        _ => {
                            info!(
                                "Executing tool call: {} with arguments: {:?}",
                                function_name, tool.function.arguments
                            );
                            let observation = self.base_agent.tools.call(&tool.function);
                            match observation {
                                Ok(observation) => {
                                    observations.push(format!(
                                        "Observation from {}: {}",
                                        function_name,
                                        observation.chars().take(30000).collect::<String>()
                                    ));
                                }
                                Err(e) => {
                                    observations.push(e.to_string());
                                    info!("Error: {}", e);
                                }
                            }
                        }
                    }
                }
                step_log.observations = Some(observations);

                let combined = step_log.observations.clone().unwrap_or_default().join("\n");

                info!(
                    "Observation: {} \n ....This content has been truncated due to the 30000 character limit.....",
                    combined.trim().chars().take(30000).collect::<String>()
                );

                if let Some(answer) = detect_final_answer(&combined) {
                    Ok(Some(answer))
                } else {
                    Ok(None)
                }
            }
            Step::ToolCall(tool_call) => {
                let function_name = tool_call.function.name.clone();
                info!("Executing tool call: {}", function_name);
                let observation = self.base_agent.tools.call(&tool_call.function)?;
                info!("Observation: {}", truncate_observation(&observation, 30000));
                return Ok(Some(observation));
            }
            Step::PlanningStep(plan, facts) => {
                info!("Plan:\n{}", truncate_observation(plan, 30000));
                info!("Facts:\n{}", truncate_observation(facts, 30000));
                return Ok(None);
            }
            Step::TaskStep(task) => {
                info!("Task: {}", task);
                return Ok(None);
            }
            Step::SystemPromptStep(prompt) => {
                info!("System prompt: {}", truncate_observation(prompt, 30000));
                return Ok(None);
            }
        }
    }

    fn stream_run(&mut self, task: &str) -> Result<String> {
        let mut final_answer: Option<String> = None;
        while final_answer.is_none() && self.get_step_number() < self.get_max_steps() {
            println!("Step number: {:?}", self.get_step_number());
            let mut step_log = Step::ActionStep(AgentStep {
                agent_memory: None,
                llm_output: None,
                tool_call: None,
                error: None,
                observations: None,
                _step: self.get_step_number(),
            });
            final_answer = self.step_stream(&mut step_log, &mut |t| print!("{}", t))?;
            self.get_logs_mut().push(step_log);
            self.increment_step_number();
        }

        if final_answer.is_none() && self.get_step_number() >= self.get_max_steps() {
            final_answer = self.provide_final_answer(task)?;
        }
        info!(
            "Final answer: {}",
            final_answer
                .clone()
                .unwrap_or("Could not find answer".to_string())
        );
        Ok(final_answer.unwrap_or_else(|| "Max steps reached without final answer".to_string()))
    }
}

#[cfg(feature = "code-agent")]
pub struct CodeAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
    local_python_interpreter: LocalPythonInterpreter,
}

#[cfg(feature = "code-agent")]
impl<M: Model> CodeAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AnyTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(CODE_SYSTEM_PROMPT);

        let base_agent = MultiStepAgent::new(
            model,
            tools,
            Some(system_prompt),
            managed_agents,
            description,
            max_steps,
        )?;
        let local_python_interpreter = LocalPythonInterpreter::new(
            base_agent
                .tools
                .iter()
                .map(|tool| tool.clone_box())
                .collect(),
        );

        Ok(Self {
            base_agent,
            local_python_interpreter,
        })
    }
}

#[cfg(feature = "code-agent")]
impl<M: Model + Debug> Agent for CodeAgent<M> {
    fn name(&self) -> &'static str {
        self.base_agent.name()
    }
    fn get_max_steps(&self) -> usize {
        self.base_agent.get_max_steps()
    }
    fn get_step_number(&self) -> usize {
        self.base_agent.get_step_number()
    }
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number()
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }
    fn reset_step_number(&mut self) {
        self.base_agent.reset_step_number()
    }
    fn set_task(&mut self, task: &str) {
        self.base_agent.set_task(task);
    }
    fn get_system_prompt(&self) -> &str {
        self.base_agent.get_system_prompt()
    }
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        let result = match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory);

                let llm_output = self.base_agent.model.run(
                    self.base_agent.input_messages.as_ref().unwrap().clone(),
                    vec![],
                    None,
                    Some(HashMap::from([(
                        "stop".to_string(),
                        vec!["Observation:".to_string(), "<end_code>".to_string()],
                    )])),
                )?;

                let response = llm_output.get_response()?;
                step_log.llm_output = Some(response.clone());

                let code = parse_code_blobs(&response).map_err(|e| {
                    step_log.error = Some(e.clone());
                    info!("Error: {}\n{}", response, e);
                    anyhow::anyhow!(e)
                })?;

                info!("Code: {}", code);
                step_log.tool_call = Some(vec![ToolCall {
                    id: None,
                    call_type: Some("function".to_string()),
                    function: FunctionCall {
                        name: "python_interpreter".to_string(),
                        arguments: serde_json::json!({ "code": code }),
                    },
                }]);
                let result = self.local_python_interpreter.forward(&code);
                match result {
                    Ok(result) => {
                        let (result, execution_logs) = result;
                        let mut observation = if !execution_logs.is_empty() {
                            format!("Execution logs: {}", execution_logs)
                        } else {
                            format!("Observation: {}", result)
                        };
                        if let Some(answer) = detect_final_answer(&observation) {
                            return Ok(Some(answer));
                        }
                        if observation.len() > 30000 {
                            observation = truncate_observation(&observation, 30000);
                        }
                        info!("Observation: {}", observation);

                        step_log.observations = Some(vec![observation]);
                    }
                    Err(e) => match e {
                        InterpreterError::FinalAnswer(answer) => {
                            return Ok(Some(answer));
                        }
                        _ => {
                            step_log.error = Some(AgentError::Execution(e.to_string()));
                            info!("Error: {}", e);
                        }
                    },
                }
                Ok(None)
            }
            Step::ToolCall(tool_call) => {
                let function_name = tool_call.function.name.clone();
                info!("Executing tool call: {}", function_name);
                let observation = self.base_agent.tools.call(&tool_call.function)?;
                info!("Observation: {}", truncate_observation(&observation, 30000));
                return Ok(Some(observation));
            }
            Step::PlanningStep(plan, facts) => {
                info!("Plan:\n{}", truncate_observation(plan, 30000));
                info!("Facts:\n{}", truncate_observation(facts, 30000));
                return Ok(None);
            }
            Step::TaskStep(task) => {
                info!("Task: {}", task);
                return Ok(None);
            }
            Step::SystemPromptStep(prompt) => {
                info!("System prompt: {}", truncate_observation(prompt, 30000));
                return Ok(None);
            }
        };

        result
    }
}

#[cfg(feature = "code-agent")]
pub fn parse_code_blobs(code_blob: &str) -> Result<String, AgentError> {
    let pattern = r"```(?:py|python)?\n([\s\S]*?)\n```";
    let re = Regex::new(pattern).map_err(|e| AgentError::Execution(e.to_string()))?;

    let matches: Vec<String> = re
        .captures_iter(code_blob)
        .map(|cap| cap[1].trim().to_string())
        .collect();

    if matches.is_empty() {
        // Check if it's a direct code blob or final answer
        if code_blob.contains("final") && code_blob.contains("answer") {
            return Err(AgentError::Parsing(
                "The code blob is invalid. It seems like you're trying to return the final answer. Use:\n\
                Code:\n\
                ```py\n\
                final_answer(\"YOUR FINAL ANSWER HERE\")\n\
                ```".to_string(),
            ));
        }

        return Err(AgentError::Parsing(
            "The code blob is invalid. Make sure to include code with the correct pattern, for instance:\n\
            Thoughts: Your thoughts\n\
            Code:\n\
            ```py\n\
            # Your python code here\n\
            ```".to_string(),
        ));
    }

    Ok(matches.join("\n\n"))
}

/// Try to detect a final answer in the given text.
///
/// It looks for patterns like `Final Answer: foo` or `final_answer("foo")` and
/// returns the captured answer if found.
pub fn detect_final_answer(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    if let Some(start) = lower.find("final answer:") {
        let answer = text[start + 13..].trim();
        if !answer.is_empty() {
            return Some(answer.to_string());
        }
    }
    if let Some(start) = lower.find("final_answer(") {
        let remainder = &text[start + 13..];
        if let Some(end) = remainder.find(')') {
            let answer = remainder[..end].trim().trim_matches('"');
            if !answer.is_empty() {
                return Some(answer.to_string());
            }
        }
    }
    None
}

/// Truncate an observation string while trying to keep both the beginning and
/// end. Returns the truncated string.
pub fn truncate_observation(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let half = limit / 2;
    let start: String = text.chars().take(half).collect();
    let end: String = text
        .chars()
        .rev()
        .take(half)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{} ...[truncated]... {}", start, end)
}

/// An agent that first generates a high level plan and then executes each plan
/// step using a `FunctionCallingAgent`.
pub struct PlanningAgent<M: Model + Clone> {
    planner: MultiStepAgent<M>,
    executor: FunctionCallingAgent<M>,
    logs: Vec<Step>,
}

impl<M: Model + Debug + Clone> PlanningAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AnyTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let planner_tools = tools.iter().map(|t| t.clone_box()).collect();
        let planner = MultiStepAgent::new(
            model.clone(),
            planner_tools,
            None,
            None,
            description,
            max_steps,
        )?;
        let executor = FunctionCallingAgent::new(
            model,
            tools,
            system_prompt,
            managed_agents,
            description,
            max_steps,
        )?;
        Ok(Self {
            planner,
            executor,
            logs: Vec::new(),
        })
    }

    fn parse_plan(plan: &str) -> Vec<String> {
        plan.lines()
            .filter_map(|l| {
                let trimmed = l.trim();
                if trimmed.is_empty() || trimmed.starts_with("<end_plan>") {
                    None
                } else if trimmed
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
                {
                    let step = trimmed
                        .trim_start_matches(|c: char| c.is_ascii_digit())
                        .trim_start_matches(['.', ')', '-', ' '].as_ref())
                        .to_string();
                    Some(step)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_final_answer_colon() {
        let text = "Some text. Final Answer: 42";
        assert_eq!(detect_final_answer(text), Some("42".to_string()));
    }

    #[test]
    fn test_detect_final_answer_fn() {
        let text = "ignored final_answer(\"hello\") trailing";
        assert_eq!(detect_final_answer(text), Some("hello".to_string()));
    }

    #[test]
    fn test_truncate_observation() {
        let text = "a".repeat(35000);
        let truncated = truncate_observation(&text, 30000);
        assert!(truncated.len() < text.len());
        assert!(truncated.contains("truncated"));
    }
}

impl<M: Model + Debug + Clone> Agent for PlanningAgent<M> {
    fn name(&self) -> &'static str {
        "PlanningAgent"
    }
    fn get_max_steps(&self) -> usize {
        self.executor.get_max_steps()
    }
    fn get_step_number(&self) -> usize {
        self.executor.get_step_number()
    }
    fn reset_step_number(&mut self) {
        self.executor.reset_step_number();
    }
    fn increment_step_number(&mut self) {
        self.executor.increment_step_number();
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        &mut self.logs
    }
    fn set_task(&mut self, task: &str) {
        self.planner.set_task(task);
        self.executor.set_task(task);
    }
    fn get_system_prompt(&self) -> &str {
        self.executor.get_system_prompt()
    }
    fn model(&self) -> &dyn Model {
        self.executor.model()
    }
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        self.executor.step(log_entry)
    }
    fn run(&mut self, task: &str, stream: bool, reset: bool) -> Result<String> {
        if reset {
            self.logs.clear();
        }
        self.set_task(task);
        self.planner.planning_step(task, true, 0);
        if let Some(Step::PlanningStep(plan, facts)) = self.planner.logs.last().cloned() {
            self.logs.push(Step::PlanningStep(plan.clone(), facts));
            let steps = Self::parse_plan(&plan);
            let mut final_answer = String::new();
            for step_task in steps {
                final_answer = self.executor.run(&step_task, stream, true)?;
                self.logs.extend(self.executor.get_logs_mut().drain(..));
            }
            Ok(final_answer)
        } else {
            Err(anyhow::anyhow!("Failed to generate plan"))
        }
    }
}

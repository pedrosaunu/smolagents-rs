use std::collections::HashMap;

use crate::{
    errors::AgentError,
    models::{openai::ToolCall, types::Message},
    tools::tool_traits::ToolInfo,
};
use anyhow::Result;
pub trait ModelResponse {
    fn get_response(&self) -> Result<String, AgentError>;
    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError>;
}

pub trait Model {
    fn run(
        &self,
        input_messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError>;

    fn run_stream(
        &self,
        input_messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        callback: &mut dyn FnMut(&str),
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let response = self.run(input_messages, tools, max_tokens, args)?;
        let text = response.get_response()?;
        callback(&text);
        Ok(response)
    }
}

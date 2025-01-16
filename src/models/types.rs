use std::fmt::Debug;
use serde::{Deserialize, Serialize};


#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    #[serde(rename = "tool")]
    ToolCall,
    #[serde(rename = "tool_response")]
    ToolResponse,
}

#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}



use std::collections::HashMap;

use reqwest::blocking::Client;
use serde::Serialize;
use anyhow::Result;
use serde_json::json;
use std::fmt::Debug;
use serde::Deserialize;

use crate::{errors::AgentError, tools::{get_json_schema, Tool}};

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    #[serde(rename = "tool")]
    ToolCall,
    #[serde(rename = "tool_response")]
    ToolResponse
}

#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String
}

pub trait Model {
    fn run(
        &self,
        input_messages: Vec<Message>,
        tools: Vec<&Box<dyn Tool>>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>
    ) -> Result<impl ModelResponse, AgentError>;
}

pub trait ModelResponse {
    fn get_response(&self) -> Result<String>;
    fn get_tools_used(&self) -> Result<Vec<ToolCall>>;
}


#[derive(Debug)]
pub struct OpenAIServerModel {
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
}

impl OpenAIServerModel {
    pub fn new(model_id: Option<&str>, temperature: Option<f32>, api_key: Option<String>) -> Self {
        let api_key = match api_key {
            Some(key) => key,
            None => std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set"),
        };
        let model_id = model_id.unwrap_or("gpt-4o-mini");

        let client = reqwest::blocking::Client::new();
        OpenAIServerModel {
            model_id: model_id.to_string(),
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
        }
    }

}

impl Model for OpenAIServerModel {
    fn run(&self, messages: Vec<Message>, tools_to_call_from: Vec<&Box<dyn Tool>>, max_tokens: Option<usize>, args: Option<HashMap<String, Vec<String>>>) -> Result<impl ModelResponse, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);
        // messages.iter().for_each(|message| println!("Message: {}", message.content));
        let messages = messages.iter().map(|message| {
            json!({
                "role": message.role,
                "content": message.content
            })
        }).collect::<Vec<_>>();

        let tools = tools_to_call_from.into_iter()
            .map(|tool| get_json_schema(tool))
            .collect::<Vec<_>>();
        let body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            "max_tokens": max_tokens,
            "tool_choice": "required"
        });

        if let Some(args) = args {
            let mut body = body.as_object().unwrap().clone();
            for (key, value) in args {
                body.insert(key, json!(value));
            }
        }


        let response = self.client.post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", self.api_key))
        .json(&body)
        .send()
        .map_err(|e| AgentError::Generation(format!("Failed to get response from OpenAI: {}", e.to_string())))?;

        match response.status() {
            reqwest::StatusCode::OK => {
                Ok(response.json::<OpenAIResponse>().unwrap())
            }
            _ => {
                Err(AgentError::Generation(format!("Failed to get response from OpenAI: {}", response.text().unwrap())))
            }
        }

    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAIResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl FunctionCall {
    pub fn get_arguments(&self) -> Result<HashMap<String, String>> {
        Ok(serde_json::from_str(&self.arguments)?)
    }
}

impl ModelResponse for OpenAIResponse {
    fn get_response(&self) -> Result<String> {
        Ok(self.choices.first().unwrap().message.content.clone().unwrap_or_default())
    }
    fn get_tools_used(&self) -> Result<Vec<ToolCall>> {
        Ok(self.choices.first().unwrap().message.tool_calls.as_ref().unwrap_or(&vec![]).clone())
    }
}

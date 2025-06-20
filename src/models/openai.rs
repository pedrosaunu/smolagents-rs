use std::collections::HashMap;

use crate::errors::AgentError;
use crate::models::model_traits::{Model, ModelResponse};
use crate::models::types::{Message, MessageRole};
use crate::tools::ToolInfo;
use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    #[serde(deserialize_with = "deserialize_arguments")]
    pub arguments: Value,
}

// Add this function to handle argument deserialization
fn deserialize_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;

    // If it's a string, try to parse it as JSON
    if let Value::String(s) = &value {
        if let Ok(parsed) = serde_json::from_str(s) {
            return Ok(parsed);
        }
    }

    Ok(value)
}

impl FunctionCall {
    pub fn get_arguments(&self) -> Result<HashMap<String, String>> {
        // First try to parse as a HashMap directly
        if let Ok(map) = serde_json::from_value(self.arguments.clone()) {
            return Ok(map);
        }

        // If that fails, try to parse as a string and then parse that string as JSON
        if let Value::String(arg_str) = &self.arguments {
            if let Ok(parsed) = serde_json::from_str(arg_str) {
                return Ok(parsed);
            }
        }

        // If all parsing attempts fail, return the original error
        Err(anyhow::anyhow!(
            "Failed to parse arguments as HashMap or JSON string"
        ))
    }
}

impl ModelResponse for OpenAIResponse {
    fn get_response(&self) -> Result<String, AgentError> {
        Ok(self
            .choices
            .first()
            .ok_or(AgentError::Generation(
                "No message returned from OpenAI".to_string(),
            ))?
            .message
            .content
            .clone()
            .unwrap_or_default())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError> {
        Ok(self
            .choices
            .first()
            .ok_or(AgentError::Generation(
                "No message returned from OpenAI".to_string(),
            ))?
            .message
            .tool_calls
            .clone()
            .unwrap_or_default())
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIServerModel {
    pub base_url: String,
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
}

impl OpenAIServerModel {
    pub fn new(
        base_url: Option<&str>,
        model_id: Option<&str>,
        temperature: Option<f32>,
        api_key: Option<String>,
    ) -> Self {
        let api_key = api_key.unwrap_or_else(|| {
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set")
        });
        let model_id = model_id.unwrap_or("gpt-4o-mini").to_string();
        let base_url = base_url.unwrap_or("https://api.openai.com/v1/chat/completions");
        let client = Client::new();

        OpenAIServerModel {
            base_url: base_url.to_string(),
            model_id,
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
        }
    }
}

impl Model for OpenAIServerModel {
    fn run(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);

        let messages = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message.content
                })
            })
            .collect::<Vec<_>>();
        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        });

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("required");
        }

        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from OpenAI: {}", e))
            })?;

        match response.status() {
            reqwest::StatusCode::OK => {
                let response = response.json::<OpenAIResponse>().unwrap();
                Ok(Box::new(response))
            }
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from OpenAI: {}",
                response.text().unwrap()
            ))),
        }
    }

    fn run_stream(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        callback: &mut dyn FnMut(&str),
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);

        let messages = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message.content
                })
            })
            .collect::<Vec<_>>();
        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": true
        });

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("required");
        }

        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from OpenAI: {}", e))
            })?;

        use std::io::{BufRead, BufReader};

        let mut reader = BufReader::new(response);
        let mut content = String::new();
        let mut line = String::new();
        while reader.read_line(&mut line).map_err(|e| AgentError::Generation(e.to_string()))? > 0 {
            if line.starts_with("data: ") {
                let data = line.trim_start_matches("data: ").trim();
                if data == "[DONE]" {
                    break;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(token) = val["choices"][0]["delta"]["content"].as_str() {
                        callback(token);
                        content.push_str(token);
                    }
                }
            }
            line.clear();
        }

        let response = OpenAIResponse {
            choices: vec![Choice {
                message: AssistantMessage {
                    role: MessageRole::Assistant,
                    content: Some(content),
                    tool_calls: None,
                    refusal: None,
                },
            }],
        };
        Ok(Box::new(response))
    }
}

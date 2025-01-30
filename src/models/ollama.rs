use std::collections::HashMap;

use serde::Deserialize;
use serde_json::json;

use anyhow::Result;
use crate::{
    errors::AgentError,
    tools::{get_json_schema, Tool},
};

use super::{
    model_traits::{Model, ModelResponse}, openai::ToolCall, types::{Message, MessageRole}
};

#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    pub message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ModelResponse for OllamaResponse {
    fn get_response(&self) -> Result<String> {
        Ok(self.message.content.clone().unwrap_or_default())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>> {
        Ok(self.message.tool_calls.clone().unwrap_or_default())
    }
}

#[derive(Debug, Clone)]
pub struct OllamaModel {
    model_id: String,
    temperature: f32,
    url: String,
    client: reqwest::blocking::Client,
    ctx_length: usize,
}

#[derive(Default)]
pub struct OllamaModelBuilder {
    model_id: String,
    temperature: Option<f32>,
    client: Option<reqwest::blocking::Client>,
    url: Option<String>,
    ctx_length: Option<usize>,
}

impl OllamaModelBuilder {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::new();
        Self {
            model_id: "llama3.2".to_string(),
            temperature: Some(0.1),
            client: Some(client),
            url: Some("http://localhost:11434".to_string()),
            ctx_length: Some(2048),
        }
    }

    pub fn model_id(mut self, model_id: &str) -> Self {
        self.model_id = model_id.to_string();
        self
    }

    pub fn temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn url(mut self, url: String) -> Self {
        self.url = Some(url);
        self
    }

    pub fn ctx_length(mut self, ctx_length: usize) -> Self {
        self.ctx_length = Some(ctx_length);
        self
    }

    pub fn build(self) -> OllamaModel {
        OllamaModel {
            model_id: self.model_id,
            temperature: self.temperature.unwrap_or(0.1),
            url: self.url.unwrap_or("http://localhost:11434".to_string()),
            client: self.client.unwrap_or_default(),
            ctx_length: self.ctx_length.unwrap_or(2048),
        }
    }
}

impl Model for OllamaModel {
    fn run(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<Box<&dyn Tool>>,
        max_tokens: Option<usize>,
        _args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<impl ModelResponse, AgentError> {
        let messages = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message.content
                })
            })
            .collect::<Vec<_>>();

        let tools = tools_to_call_from
            .into_iter()
            .map(|tool| get_json_schema(&**tool))
            .collect::<Vec<_>>();

        let body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "stream": false,
            "options": json!({
                "num_ctx": self.ctx_length
            }),
            "tools": tools,
            "max_tokens": max_tokens.unwrap_or(1500),
        });
        // println!("Body: {}", serde_json::to_string_pretty(&body.get("messages").unwrap()).unwrap());
        let response = self.client.post(format!("{}/api/chat", self.url)).json(&body).send().map_err(|e| {
            AgentError::Generation(format!("Failed to get response from Ollama: {}", e))
        })?;
        Ok(response.json::<OllamaResponse>().unwrap())
    }
}
